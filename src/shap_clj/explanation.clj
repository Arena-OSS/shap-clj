(ns shap-clj.explanation
  "The Explanation value produced by every explainer.  Mirrors the
  fields of `shap.Explanation` but uses native Clojure / primitive
  array storage."
  (:require [shap-clj.coerce :as c]))

(set! *warn-on-reflection* true)

(defrecord Explanation [values base-values data feature-names prediction method extra])

(defn- filled-double-array ^"[D" [^long n ^double v]
  (let [a (double-array n)]
    (java.util.Arrays/fill a v)
    a))

(defn explanation
  "Construct an Explanation from named inputs."
  [{:keys [values base-values data feature-names prediction method extra]
    :or {method :unknown extra {}}}]
  (let [V (c/to-2d values)
        D (c/to-2d data)
        n (c/n-rows V)
        d (c/n-cols V)
        bv (cond
             (nil? base-values) (double-array n)
             (number? base-values) (filled-double-array n (double base-values))
             :else (c/to-1d base-values))
        feature-names (or feature-names
                          (vec (map #(str "f" %) (range d))))]
    (when (not= d (count feature-names))
      (throw (ex-info "feature-names length must equal #columns"
                      {:ncols d :n-names (count feature-names)})))
    (->Explanation V bv D (vec feature-names) prediction method extra)))

(defn row
  "Return a single-row Explanation for sample i."
  [expl i]
  (let [V (:values expl)
        D (:data expl)
        bv (:base-values expl)
        pred (:prediction expl)
        i (int i)]
    (explanation
      {:values [(vec (aget ^"[[D" V i))]
       :base-values (double-array 1 (aget ^"[D" bv i))
       :data [(vec (aget ^"[[D" D i))]
       :feature-names (:feature-names expl)
       :prediction (when pred (double-array 1 (aget ^"[D" pred i)))
       :method (:method expl)
       :extra (:extra expl)})))

(defn mean-abs-values
  "Vector of mean |SHAP value| per feature across all samples."
  ^"[D" [expl]
  (let [V (:values expl)
        V ^"[[D" V
        n (alength V)
        d (alength ^"[D" (aget V 0))
        out (double-array d)]
    (dotimes [i n]
      (let [r ^"[D" (aget V i)]
        (dotimes [j d]
          (aset out j (+ (aget out j) (Math/abs (aget r j)))))))
    (dotimes [j d]
      (aset out j (/ (aget out j) (double n))))
    out))

(defn feature-ranking
  "Return a vector of {:feature :mean-abs-value} sorted descending by importance."
  [expl]
  (let [mabs (mean-abs-values expl)
        names (:feature-names expl)]
    (->> (map-indexed (fn [i n] {:feature n :mean-abs-value (aget ^"[D" mabs i)}) names)
         (sort-by :mean-abs-value >)
         vec)))

(defn efficiency-residuals
  "Per-sample Σ_j φ_j + base − prediction.  Valid SHAP explanations
  keep this ≈ 0."
  ^"[D" [expl]
  (let [V ^"[[D" (:values expl)
        n (alength V)
        d (alength ^"[D" (aget V 0))
        bv ^"[D" (:base-values expl)
        pred ^"[D" (:prediction expl)
        out (double-array n)]
    (when (nil? pred)
      (throw (ex-info "efficiency-residuals requires :prediction" {})))
    (dotimes [i n]
      (let [r ^"[D" (aget V i)
            s (loop [j 0 acc 0.0]
                (if (= j d)
                  acc
                  (recur (inc j) (+ acc (aget r j)))))]
        (aset out i (- (+ s (aget bv i)) (aget pred i)))))
    out))
