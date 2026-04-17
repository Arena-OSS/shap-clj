(ns shap-clj.shap
  "Top-level, user-facing API for the shap-clj SHAP library.

  Typical usage:

    (require '[shap-clj.shap :as shap])

    ;; A model is any (fn [X] predictions).  X is an (n × d) matrix
    ;; and `predictions` a length-n numeric vector / double-array.
    (def model (fn [X] (map #(+ (* 2 (first %)) (last %)) X)))

    (def bg      (shap/background full-training-matrix 100))
    (def x-rows  [[1.0 2.0 3.0] [0.5 1.1 -0.3]])

    ;; Pick an explainer:
    (shap/explain model x-rows {:background bg :method :kernel})
    (shap/explain model x-rows {:background bg :method :permutation})
    (shap/explain model x-rows {:background bg :method :exact})
    (shap/explain model x-rows {:background bg :method :linear})

  Each call returns a `shap-clj.explanation.Explanation` with the
  SHAP values, base values, and the originating data."
  (:require [shap-clj.coerce :as c]
            [shap-clj.explainer.exact :as exact]
            [shap-clj.explainer.kernel :as kernel]
            [shap-clj.explainer.linear :as linear]
            [shap-clj.explainer.permutation :as permutation]
            [shap-clj.explanation :as explanation]
            [shap-clj.masker :as masker]
            [shap-clj.model :as model]))

(set! *warn-on-reflection* true)

;; ---------------------------------------------------------------------------
;; Sampling / background helpers
;; ---------------------------------------------------------------------------

(defn background
  "Sample or down-select `n` rows from a data set for use as the
  background distribution.  If the dataset already has ≤ n rows, it
  is returned as-is.  Uses a seeded RNG for reproducibility."
  ([data n] (background data n 42))
  ([data n seed]
   (let [X (c/to-2d data)
         rows (c/n-rows X)]
     (if (<= rows (long n))
       X
       (let [rng (java.util.Random. (long seed))
             idx (loop [acc (transient #{}) guard 0]
                   (cond
                     (= (count acc) (long n)) (persistent! acc)
                     (> guard (* 20 rows)) (persistent! acc)
                     :else
                     (recur (conj! acc (.nextInt rng rows)) (inc guard))))
             out (make-array Double/TYPE (count idx) (c/n-cols X))]
         (doseq [[i src-i] (map-indexed vector (seq idx))]
           (System/arraycopy ^"[D" (c/row X src-i) 0
                             ^"[D" (aget ^"[[D" out i) 0
                             (c/n-cols X)))
         out)))))

;; ---------------------------------------------------------------------------
;; Main entry point
;; ---------------------------------------------------------------------------

(defn explain
  "Explain the predictions of `model` on rows `X`.

  Options (keyword args or option map):
    :background     required for all non-trivial cases (data matrix)
    :method         :kernel (default) | :permutation | :exact | :linear
    :n-samples      KernelSHAP: coalitions per row       (default 2·d + 2048)
    :n-permutations PermutationExplainer: permutations   (default 10)
    :coefs :intercept  LinearExplainer: supply explicit coefficients
    :feature-names  vector<string>
    :seed           RNG seed                              (default 42)"
  ([model X] (explain model X {}))
  ([model X {:keys [background method] :or {method :kernel} :as opts}]
   (when (and (not= method :linear-from-params) (nil? background))
     (throw (ex-info "explain requires :background"
                     {:method method})))
   (let [m (model/as-model model)]
     (case method
       :kernel      (kernel/explain m background X opts)
       :permutation (permutation/explain m background X opts)
       :exact       (exact/explain m background X opts)
       :linear      (linear/explain m background X opts)
       (throw (ex-info "Unknown :method" {:method method
                                          :known [:kernel :permutation :exact :linear]}))))))

;; ---------------------------------------------------------------------------
;; Convenience constructors — give a reusable "explainer" closure
;; ---------------------------------------------------------------------------

(defn kernel-explainer
  "Return a function `(explainer X & [opts])` that computes KernelSHAP
  values against a fixed model + background."
  [model background & [base-opts]]
  (let [m (model/as-model model)]
    (fn
      ([X] (kernel/explain m background X (or base-opts {})))
      ([X opts] (kernel/explain m background X (merge base-opts opts))))))

(defn permutation-explainer
  [model background & [base-opts]]
  (let [m (model/as-model model)]
    (fn
      ([X] (permutation/explain m background X (or base-opts {})))
      ([X opts] (permutation/explain m background X (merge base-opts opts))))))

(defn exact-explainer
  [model background & [base-opts]]
  (let [m (model/as-model model)]
    (fn
      ([X] (exact/explain m background X (or base-opts {})))
      ([X opts] (exact/explain m background X (merge base-opts opts))))))

(defn linear-explainer
  [model background & [base-opts]]
  (let [m (model/as-model model)]
    (fn
      ([X] (linear/explain m background X (or base-opts {})))
      ([X opts] (linear/explain m background X (merge base-opts opts))))))

;; ---------------------------------------------------------------------------
;; Re-exports for ergonomic access
;; ---------------------------------------------------------------------------

(def as-model model/as-model)
(def predict  model/predict)
(def row      explanation/row)
(def mean-abs-values  explanation/mean-abs-values)
(def feature-ranking  explanation/feature-ranking)
(def efficiency-residuals explanation/efficiency-residuals)
