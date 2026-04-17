(ns shap-clj.plot
  "Minimal visualisation helpers.  Each function returns a plain
  Clojure data map suitable for rendering with Hanami / Plotly /
  Clay (or for inspection in a REPL).

  We intentionally do NOT take a dependency on any plotting library
  so this namespace loads in a headless CI.  The returned maps
  follow the Plotly JS schema which Clay + noj render natively."
  (:require [shap-clj.explanation :as expl]))

(set! *warn-on-reflection* true)

(defn bar
  "Global importance bar plot: mean |SHAP value| per feature, sorted.
  Returns a Plotly spec."
  [explanation]
  (let [ranking (expl/feature-ranking explanation)]
    {:data [{:type "bar"
             :orientation "h"
             :x (mapv :mean-abs-value ranking)
             :y (mapv :feature ranking)}]
     :layout {:title "SHAP — mean |value| per feature"
              :margin {:l 180}
              :yaxis {:autorange "reversed"}
              :xaxis {:title "mean(|SHAP value|)"}}}))

(defn waterfall
  "Waterfall plot for a single row of the explanation.  Shows
  base-value → feature contributions → prediction."
  [explanation row-index]
  (let [V ^"[[D" (:values explanation)
        data ^"[[D" (:data explanation)
        bv (aget ^"[D" (:base-values explanation) (int row-index))
        values (vec (aget V (int row-index)))
        feats (:feature-names explanation)
        pairs (->> (map vector feats values (aget data (int row-index)))
                   (sort-by (fn [[_ v _]] (- (Math/abs (double v))))))
        labels (mapv (fn [[f _ x]] (str f " = " x)) pairs)
        contribs (mapv (fn [[_ v _]] v) pairs)
        pred (when-let [p (:prediction explanation)]
               (aget ^"[D" p (int row-index)))]
    {:data [{:type "waterfall"
             :orientation "h"
             :x contribs
             :y labels
             :base bv
             :measure (vec (repeat (count contribs) "relative"))}]
     :layout {:title (str "SHAP waterfall — row " row-index
                          (when pred (format " (pred %.4f)" (double pred))))
              :margin {:l 260}
              :yaxis {:autorange "reversed"}}}))

(defn beeswarm
  "Simplified beeswarm: for every (sample, feature) pair, a point
  positioned at (shap-value, feature) with jitter in y and color ∝ feature value."
  [explanation]
  (let [V ^"[[D" (:values explanation)
        D ^"[[D" (:data explanation)
        n (alength V)
        d (alength ^"[D" (aget V 0))
        feats (:feature-names explanation)
        ranking (expl/feature-ranking explanation)
        ordered (map :feature ranking)
        rng (java.util.Random. 1)
        index-of-feat (into {} (map-indexed (fn [i f] [f i]) feats))]
    {:data (for [f ordered
                 :let [j (index-of-feat f)
                       xs (mapv #(aget ^"[D" (aget V %) j) (range n))
                       colors (mapv #(aget ^"[D" (aget D %) j) (range n))]]
             {:type "scatter"
              :mode "markers"
              :name f
              :x xs
              :y (mapv (fn [_] (+ (.indexOf ^java.util.List (vec ordered) f)
                                  (- (* 0.8 (.nextDouble rng)) 0.4)))
                       xs)
              :marker {:color colors :colorscale "RdBu" :showscale false}})
     :layout {:title "SHAP beeswarm"
              :margin {:l 180}
              :yaxis {:tickvals (range (count ordered))
                      :ticktext (vec ordered)
                      :autorange "reversed"}
              :xaxis {:title "SHAP value"}
              :showlegend false}}))

(defn force-like
  "Summary table of (feature, value, shap) tuples for a single row,
  sorted by SHAP magnitude.  A lightweight stand-in for shap.force_plot
  that returns data you can render however you want."
  [explanation row-index]
  (let [V ^"[[D" (:values explanation)
        D ^"[[D" (:data explanation)
        feats (:feature-names explanation)
        values (vec (aget V (int row-index)))
        inputs (vec (aget D (int row-index)))
        pred (when-let [p (:prediction explanation)]
               (aget ^"[D" p (int row-index)))
        base (aget ^"[D" (:base-values explanation) (int row-index))]
    {:base-value base
     :prediction pred
     :sum-of-shap (reduce + values)
     :contributions
     (->> (map (fn [f v x] {:feature f :value x :shap v}) feats values inputs)
          (sort-by (fn [m] (- (Math/abs (double (:shap m)))))))}))
