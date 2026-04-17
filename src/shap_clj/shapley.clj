(ns shap-clj.shapley
  "Core Shapley-value math: kernel weights, coalition enumeration,
  and weighted coalition sampling.

  A coalition is represented as a boolean-array of length n where
  true means `feature is present (take from instance)` and false
  means `feature is absent (take from background)`."
  (:require [clojure.math.combinatorics :as combo]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

;; ---------------------------------------------------------------------------
;; Combinatorics
;; ---------------------------------------------------------------------------

(defn binomial
  "C(n, k) as a double. For n up to a few hundred this stays well-bounded."
  ^double [^long n ^long k]
  (cond
    (or (neg? k) (> k n)) 0.0
    (or (zero? k) (= k n)) 1.0
    :else
    (let [k (long (min k (- n k)))]
      (loop [i 1 acc 1.0]
        (if (> i k)
          acc
          (recur (inc i)
                 (double (/ (* acc (double (- n (- i 1))))
                            (double i)))))))))

;; ---------------------------------------------------------------------------
;; Shapley kernel
;; ---------------------------------------------------------------------------

(defn shapley-kernel-weight
  "Kernel weight for a coalition of size s out of n features.
  Defined for 0 < s < n.  For s = 0 or s = n the weight is infinite
  (enforced via an equality constraint elsewhere)."
  ^double [^long n ^long s]
  (let [n (double n)
        s (double s)]
    (/ (- n 1.0)
       (* (binomial (long n) (long s)) s (- n s)))))

(defn size-kernel-weight
  "Weight placed on *all* coalitions of size s together, i.e.
  C(n,s) · π(s).  Used when we choose a coalition size and then
  sample a coalition uniformly within it."
  ^double [^long n ^long s]
  (let [n (double n) s (double s)]
    (/ (- n 1.0) (* s (- n s)))))

;; ---------------------------------------------------------------------------
;; Coalition helpers
;; ---------------------------------------------------------------------------

(defn indices->mask ^booleans [^long n indices]
  (let [m (boolean-array n false)]
    (doseq [i indices] (aset m (int i) true))
    m))

(defn mask->indices [^booleans m]
  (let [n (alength m)]
    (loop [i 0 acc (transient [])]
      (if (= i n)
        (persistent! acc)
        (recur (inc i)
               (if (aget m i) (conj! acc i) acc))))))

(defn mask-complement ^booleans [^booleans m]
  (let [n (alength m)
        out (boolean-array n)]
    (dotimes [i n] (aset out i (not (aget m i))))
    out))

(defn mask-size ^long [^booleans m]
  (let [n (alength m)]
    (loop [i 0 c 0]
      (if (= i n)
        c
        (recur (inc i) (if (aget m i) (inc c) c))))))

;; ---------------------------------------------------------------------------
;; Full enumeration of all 2^n - 2 non-trivial coalitions
;; ---------------------------------------------------------------------------

(defn enumerate-coalitions
  "Return a vector of all 2^n - 2 non-trivial coalitions (sizes 1..n-1)
  each as a boolean-array of length n, along with their kernel weights.
  Output: {:coalitions [bool-arrays...] :weights double-array :sizes int-array}."
  [^long n]
  (let [pairs (for [s (range 1 n)
                    combo (combo/combinations (range n) s)]
                [(indices->mask n combo) s])
        M (count pairs)
        ws (double-array M)
        sizes (int-array M)]
    (dotimes [i M]
      (let [[_ s] (nth pairs i)]
        (aset ws i (shapley-kernel-weight n s))
        (aset sizes i (int s))))
    {:coalitions (mapv first pairs)
     :weights ws
     :sizes sizes}))

;; ---------------------------------------------------------------------------
;; Weighted sampling of coalitions via size-then-indices
;; ---------------------------------------------------------------------------

(defn- sample-size
  "Draw a subset size s ∈ 1..n-1 from distribution ∝ size-kernel-weight."
  ^long [^long n ^java.util.Random rng]
  (let [weights (double-array (dec n))
        _ (dotimes [i (dec n)]
            (aset weights i (size-kernel-weight n (inc i))))
        total (double (areduce weights i s 0.0 (+ s (aget weights i))))
        r (* total (.nextDouble rng))]
    (loop [i 0 acc 0.0]
      (if (>= i (dec n))
        (dec n)
        (let [acc' (+ acc (aget weights i))]
          (if (>= acc' r)
            (inc i)
            (recur (inc i) acc')))))))

(defn- sample-indices
  "Sample k distinct indices in [0, n) uniformly."
  [^long n ^long k ^java.util.Random rng]
  (let [arr (int-array n)]
    (dotimes [i n] (aset arr i i))
    ;; Fisher-Yates partial shuffle to expose first k elements.
    (dotimes [i (long k)]
      (let [j (+ i (.nextInt rng (- n i)))
            tmp (aget arr i)]
        (aset arr i (aget arr j))
        (aset arr j tmp)))
    (let [out (int-array k)]
      (System/arraycopy arr 0 out 0 k)
      out)))

(defn sample-coalitions
  "Return `budget` coalitions sampled with Shapley-kernel weights
  (paired: each draw contributes the coalition and its complement).
  The weights returned are uniform (1.0) because the sampling itself
  already reflects the kernel; the WLS fit should use equal weights."
  ([^long n ^long budget]
   (sample-coalitions n budget (java.util.Random. 42)))
  ([^long n ^long budget ^java.util.Random rng]
   (let [half (quot budget 2)
         coalitions (transient [])
         weights (double-array budget 1.0)
         sizes (int-array budget)]
     (dotimes [p half]
       (let [s (sample-size n rng)
             idx (sample-indices n s rng)
             mask (indices->mask n idx)
             comp (mask-complement mask)]
         (conj! coalitions mask)
         (conj! coalitions comp)
         (aset sizes (* 2 p) (int s))
         (aset sizes (inc (* 2 p)) (int (- n s)))))
     (when (odd? budget)
       (let [s (sample-size n rng)
             mask (indices->mask n (sample-indices n s rng))]
         (conj! coalitions mask)
         (aset sizes (dec budget) (int s))))
     {:coalitions (persistent! coalitions)
      :weights weights
      :sizes sizes})))

;; ---------------------------------------------------------------------------
;; Choose enumerate-vs-sample based on budget
;; ---------------------------------------------------------------------------

(defn build-coalitions
  "Given n features and a sample budget, decide whether to enumerate
  (if 2^n - 2 ≤ budget) or sample.  Returns {:coalitions :weights :sizes :exact?}."
  ([^long n ^long budget]
   (build-coalitions n budget (java.util.Random. 42)))
  ([^long n ^long budget ^java.util.Random rng]
   (let [full (- (bit-shift-left 1 n) 2)]
     (if (and (< n 20) (<= full budget))
       (assoc (enumerate-coalitions n) :exact? true)
       (assoc (sample-coalitions n budget rng) :exact? false)))))
