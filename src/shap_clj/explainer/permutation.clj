(ns shap-clj.explainer.permutation
  "Permutation-based Shapley estimator.

  For a random permutation π of features, start from the fully-masked
  row (baseline), then reveal features in permutation order; each
  feature's attribution in that permutation is the change in model
  output when it is revealed.  Average over many permutations to get
  the Shapley value.

  Uses paired (antithetic) permutations — for every forward order we
  also run the reverse — which is what the reference SHAP library
  defaults to (matches the `PermutationExplainer`)."
  (:require [shap-clj.coerce :as c]
            [shap-clj.explainer.common :as common]
            [shap-clj.explanation :as expl]
            [shap-clj.masker :as masker]
            [shap-clj.model :as model]
            [shap-clj.shapley :as shap]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(defn- random-permutation
  "Fisher-Yates shuffle; returns an int-array of length n."
  [^long n ^java.util.Random rng]
  (let [arr (int-array n)]
    (dotimes [i n] (aset arr i i))
    (loop [i (dec n)]
      (when (pos? i)
        (let [j (.nextInt rng (inc i))
              tmp (aget arr i)]
          (aset arr i (aget arr j))
          (aset arr j tmp)
          (recur (dec i)))))
    arr))

(defn- reverse-permutation ^ints [^ints perm]
  (let [n (alength perm)
        rev (int-array n)]
    (dotimes [i n] (aset rev i (aget perm (- n 1 i))))
    rev))

(defn- coalitions-along
  "Given a permutation (int-array), return a seq of (d+1) coalition
  masks reflecting the cumulative sets: ∅, {π[0]}, {π[0],π[1]}, ..., N."
  [^long d ^ints perm]
  (loop [k 0
         mask (boolean-array d)
         out (transient [])]
    (if (> k d)
      (persistent! out)
      (let [m (boolean-array d)]
        (System/arraycopy mask 0 m 0 d)
        (when (pos? k) (aset m (aget perm (dec k)) true))
        (recur (inc k) m (conj! out m))))))

(defn- add-one-permutation!
  "Accumulate one permutation's contributions into `phi` (double-array)
  by running the model along the reveal path and differencing."
  [phi model masker ^"[D" x ^ints perm]
  (let [d (alength x)
        coalitions (coalitions-along d perm)
        ys (common/eval-coalitions model masker x coalitions)]
    ;; ys[0] is baseline (empty), ys[d] is full.  φ_{π[k]} += ys[k+1] - ys[k].
    (dotimes [k d]
      (let [feat (aget perm k)
            delta (- (aget ys (inc k)) (aget ys k))]
        (aset ^"[D" phi feat (+ (aget ^"[D" phi feat) delta))))))

(defn explain-row
  "PermutationExplainer attributions for one row x.  Runs `n-permutations`
  permutations (each paired with its reverse -> 2·n-permutations calls)."
  ^"[D" [model masker ^"[D" x
            {:keys [n-permutations rng]
             :or {n-permutations 10
                  rng (java.util.Random. 42)}}]
  (let [d (alength x)
        phi (double-array d)
        total-perms (* 2 (long n-permutations))]
    (dotimes [_ n-permutations]
      (let [perm (random-permutation d rng)
            rev  (reverse-permutation perm)]
        (add-one-permutation! phi model masker x perm)
        (add-one-permutation! phi model masker x rev)))
    (dotimes [i d]
      (aset phi i (/ (aget phi i) (double total-perms))))
    phi))

(defn explain
  "PermutationExplainer over every row of X.  Options:
    :n-permutations int   (default 10 -> 20 paired runs)
    :seed           long
    :feature-names  vector<string>
    :parallelism    int   (default availableProcessors; set 1 to disable)

  Rows are independent, so the per-row loop fans out across `:parallelism`
  worker threads.  Each row gets its own RNG seeded from (seed + i) so
  results stay deterministic and identical across thread counts."
  ([model background X] (explain model background X {}))
  ([model background X {:keys [n-permutations seed feature-names parallelism]
                        :or {n-permutations 10
                             seed 42
                             parallelism (.. Runtime getRuntime availableProcessors)}}]
   (let [model (model/as-model model)
         masker (masker/independent background)
         X (c/to-2d X)
         n (c/n-rows X)
         d (c/n-cols X)
         values (make-array Double/TYPE n d)
         base-val (aget (common/baseline model masker) 0)
         base-values (double-array n base-val)
         pred (model/predict model X)
         par (max 1 (int parallelism))
         run-row! (fn [^long i]
                    (let [rng (java.util.Random. (+ (long seed) i))
                          phi (explain-row model masker (aget ^"[[D" X (int i))
                                           {:n-permutations n-permutations :rng rng})]
                      (System/arraycopy phi 0 ^"[D" (aget ^"[[D" values (int i)) 0 d)))]
     (if (or (<= par 1) (<= n 1))
       (dotimes [i n] (run-row! i))
       (let [chunk-size (max 1 (long (Math/ceil (/ (double n) (double par)))))
             futs (->> (range 0 n chunk-size)
                       (mapv (fn [^long start]
                               (let [end (min n (+ start chunk-size))]
                                 (future
                                   (loop [i start]
                                     (when (< i end)
                                       (run-row! i)
                                       (recur (inc i)))))))))]
         (doseq [f futs] @f)))
     (expl/explanation {:values values
                        :base-values base-values
                        :data X
                        :feature-names feature-names
                        :prediction pred
                        :method :permutation
                        :extra {:n-permutations n-permutations
                                :seed seed
                                :parallelism par}}))))
