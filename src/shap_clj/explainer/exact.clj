(ns shap-clj.explainer.exact
  "Exact Shapley values via full enumeration of subsets.  O(2^n) in the
  number of features, so only practical for n ≲ 12.  Useful both as a
  correctness oracle for the approximate explainers and as the right
  choice for very small problems."
  (:require [clojure.math.combinatorics :as combo]
            [shap-clj.coerce :as c]
            [shap-clj.explainer.common :as common]
            [shap-clj.explanation :as expl]
            [shap-clj.masker :as masker]
            [shap-clj.model :as model]
            [shap-clj.shapley :as shap]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(defn- factorial ^double [^long n]
  (loop [i 1 acc 1.0]
    (if (> i n) acc (recur (inc i) (* acc (double i))))))

(defn- all-subset-masks
  "Return a seq of boolean-array coalitions for every subset of [0..n-1]."
  [^long n]
  (for [k (range (inc n))
        combo (combo/combinations (range n) k)]
    (shap/indices->mask n combo)))

(defn- attribution-for-feature
  "Exact φ_i for a single feature i, given a map
     `val-map`  coalition-as-index-set → f_S(x).
   Computed as
     φ_i = Σ_{S ⊆ N\\{i}} w(|S|, n) * (f_{S∪{i}} - f_S)
   with w(s, n) = s! (n-s-1)! / n!."
  ^double [^long n ^long i val-map]
  (let [n-fact (factorial n)]
    (loop [subsets (combo/subsets (remove #(= i %) (range n)))
           total 0.0]
      (if (empty? subsets)
        total
        (let [S (first subsets)
              s (count S)
              wS (/ (* (factorial s) (factorial (- n s 1))) n-fact)
              v-noi (get val-map (set S))
              v-withi (get val-map (set (conj S i)))]
          (recur (rest subsets)
                 (+ total (* wS (- (double v-withi) (double v-noi))))))))))

(defn explain-row
  "Exact Shapley values for a single row x (double-array)."
  ^"[D" [model masker ^"[D" x]
  (let [n (alength x)
        masks (all-subset-masks n)
        coalitions (vec masks)
        ys (common/eval-coalitions model masker x coalitions)
        val-map (reduce
                 (fn [m idx]
                   (assoc m (set (shap/mask->indices (nth coalitions idx)))
                          (aget ys idx)))
                 {} (range (count coalitions)))
        phi (double-array n)]
    (dotimes [i n]
      (aset phi i (attribution-for-feature n i val-map)))
    phi))

(defn explain
  "Compute exact Shapley values for every row of X, using the given
  model and background data.  Returns a shap-clj.explanation.Explanation."
  ([model background X] (explain model background X {}))
  ([model background X {:keys [feature-names]}]
   (let [model (model/as-model model)
         masker (masker/independent background)
         X (c/to-2d X)
         n (c/n-rows X)
         d (c/n-cols X)
         values (make-array Double/TYPE n d)
         base-val (aget (common/baseline model masker) 0)
         base-values (double-array n base-val)
         pred (model/predict model X)]
     (dotimes [i n]
       (let [phi (explain-row model masker (aget X i))]
         (System/arraycopy phi 0 ^"[D" (aget ^"[[D" values i) 0 d)))
     (expl/explanation {:values values
                        :base-values base-values
                        :data X
                        :feature-names feature-names
                        :prediction pred
                        :method :exact}))))
