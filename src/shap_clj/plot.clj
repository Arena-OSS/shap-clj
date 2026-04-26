(ns shap-clj.plot
  "Minimal visualisation helpers.  Each function returns a plain
  Clojure data map suitable for rendering with Hanami / Plotly /
  Clay (or for inspection in a REPL).

  We intentionally do NOT take a dependency on any plotting library
  so this namespace loads in a headless CI.  The returned maps
  follow the Plotly JS schema which Clay + noj render natively."
  (:require [clojure.string :as str]
            [shap-clj.explanation :as expl]))

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

;; ===========================================================================
;; Inline-SVG renderers — self-contained <svg> strings that embed in markdown
;; or static HTML without any JS runtime.  Each returns a string.
;; ===========================================================================

(def ^:private svg-pos-color "#c0392b") ; red — push toward class 1
(def ^:private svg-neg-color "#2c7fb8") ; blue — push toward class 0
(def ^:private svg-bar-color "#7e57c2") ; purple — neutral magnitude
(def ^:private svg-axis-color "#333")
(def ^:private svg-grid-color "#eee")
(def ^:private svg-text-color "#222")
;; Concrete font stack — we cannot rely on the markdown renderer's default
;; <text> font, especially in static markdown previewers which sometimes
;; resolve "sans-serif" to an unintended fallback or refuse to render at all.
(def ^:private svg-font-family
  "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif")
(def ^:private svg-mono-family
  "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace")
;; Default font sizes — these are SVG user units, so they scale with the SVG
;; when a markdown renderer applies max-width:100% in a narrow column. The
;; values below are tuned so a viewport as narrow as 480 CSS px (typical
;; IDE preview pane) still renders text at ≥10px on screen.
(def ^:private svg-title-fs   16)
(def ^:private svg-tick-fs    13)
(def ^:private svg-label-fs   13)
(def ^:private svg-axis-fs    13)

(defn- svg-fmt-num [^double v]
  (cond
    (Double/isNaN v) "NaN"
    (= v 0.0) "0"
    (< (Math/abs v) 0.001) (format "%.2e" v)
    (< (Math/abs v) 1.0) (format "%.3f" v)
    (< (Math/abs v) 100.0) (format "%.2f" v)
    :else (format "%.0f" v)))

(defn- svg-clamp ^double [^double v ^double lo ^double hi]
  (Math/max lo (Math/min hi v)))

(defn- svg-escape [s]
  (-> (str s)
      (str/replace "&" "&amp;")
      (str/replace "<" "&lt;")
      (str/replace ">" "&gt;")))

(defn- svg-truncate [s ^long max-len]
  (let [s (str s)]
    (if (<= (count s) max-len)
      s
      (str (subs s 0 (max 0 (- max-len 1))) "…"))))

(defn- svg-lerp-color
  "Linear interpolation between blue (low) and red (high) on t in [0,1]."
  [^double t]
  (let [t (svg-clamp t 0.0 1.0)
        ;; #2c7fb8 -> (44, 127, 184), #f7f7f7 -> (247, 247, 247), #c0392b -> (192, 57, 43)
        ;; Two-stop ramp through near-white at 0.5 for better contrast against axis
        [r1 g1 b1] [44 127 184]
        [rm gm bm] [240 240 240]
        [r2 g2 b2] [192 57 43]
        lerp (fn [^double a ^double b ^double w] (+ a (* w (- b a))))
        [r g b] (if (< t 0.5)
                  (let [w (* 2.0 t)]
                    [(lerp r1 rm w) (lerp g1 gm w) (lerp b1 bm w)])
                  (let [w (* 2.0 (- t 0.5))]
                    [(lerp rm r2 w) (lerp gm g2 w) (lerp bm b2 w)]))]
    (format "#%02x%02x%02x" (int r) (int g) (int b))))

(defn- column-of
  "Extract column j from a 2D double-array as a primitive double-array."
  ^"[D" [^"[[D" M ^long j]
  (let [n (alength M)
        out (double-array n)]
    (dotimes [i n]
      (aset out i (aget ^"[D" (aget M i) j)))
    out))

(defn- column-min-max [^"[D" col]
  (let [n (alength col)]
    (if (zero? n)
      [0.0 1.0]
      (loop [i 1
             lo (aget col 0)
             hi (aget col 0)]
        (if (= i n)
          [lo hi]
          (let [v (aget col i)]
            (recur (inc i) (Math/min lo v) (Math/max hi v))))))))

(defn- nice-tick-step
  "Return a 'nice' axis tick step that gives roughly `target` ticks across `range`."
  ^double [^double range ^long target]
  (if (or (zero? range) (Double/isNaN range))
    1.0
    (let [raw (/ range (double (max 1 target)))
          mag (Math/pow 10.0 (Math/floor (Math/log10 raw)))
          frac (/ raw mag)
          nice (cond (< frac 1.5) 1.0
                     (< frac 3.0) 2.0
                     (< frac 7.0) 5.0
                     :else 10.0)]
      (* nice mag))))

(defn- axis-ticks [^double lo ^double hi ^long target]
  (let [step (nice-tick-step (- hi lo) target)
        start (* step (Math/floor (/ lo step)))
        end (* step (Math/ceil (/ hi step)))]
    (loop [t start out []]
      (if (> t (+ end (* step 1.0e-6)))
        out
        (recur (+ t step) (conj out t))))))

(defn bar-svg
  "Horizontal bar chart of mean |SHAP| per feature.  Returns an `<svg>` string.

  Options:
    :top-n     number of features to show (default 25)
    :width     SVG width in px (default 720)
    :height    SVG height in px (default auto = top-n*22 + 80)
    :title     chart title  (default \"SHAP — mean |value| per feature\")"
  ([explanation] (bar-svg explanation {}))
  ([explanation {:keys [top-n width height title]
                 :or {top-n 25 width 600
                      title "SHAP — mean |value| per feature"}}]
   (let [ranking (expl/feature-ranking explanation)
         shown (vec (take top-n ranking))
         n (count shown)
         row-h 26
         pad-top 56
         pad-bot 50
         pad-left 220
         pad-right 80
         h (long (or height (+ pad-top pad-bot (* row-h n))))
         plot-w (- width pad-left pad-right)
         max-v (double (apply max 0.0 (map :mean-abs-value shown)))
         max-v (if (zero? max-v) 1.0 max-v)
         ticks (axis-ticks 0.0 max-v 5)
         ->x (fn [^double v] (+ pad-left (* (/ v max-v) plot-w)))
         row-y (fn [^long i] (+ pad-top (* i row-h) (/ row-h 2)))
         tick-lines (apply str
                      (for [t ticks
                            :let [x (->x t)]]
                        (str "<line x1=\"" x "\" y1=\"" pad-top
                             "\" x2=\"" x "\" y2=\"" (- h pad-bot)
                             "\" stroke=\"" svg-grid-color "\" stroke-width=\"1\"/>"
                             "<text x=\"" x "\" y=\"" (- h pad-bot -18)
                             "\" text-anchor=\"middle\" font-family=\"" svg-font-family
                             "\" font-size=\"" svg-tick-fs "\" fill=\""
                             svg-text-color "\">" (svg-fmt-num t) "</text>")))
         bars (apply str
                (map-indexed
                  (fn [i {:keys [feature mean-abs-value]}]
                    (let [y (- (row-y i) (/ row-h 2.5))
                          bar-w (- (->x (double mean-abs-value)) pad-left)
                          label-x (- pad-left 10)
                          val-x (+ pad-left bar-w 6)]
                      (str "<rect x=\"" pad-left "\" y=\"" y
                           "\" width=\"" (max 0.0 bar-w) "\" height=\"" (* row-h 0.78)
                           "\" fill=\"" svg-bar-color "\" />"
                           "<text x=\"" label-x "\" y=\"" (+ (row-y i) 5)
                           "\" text-anchor=\"end\" font-family=\"" svg-mono-family
                           "\" font-size=\"" svg-label-fs "\" fill=\""
                           svg-text-color "\">" (svg-escape (svg-truncate (str feature) 24)) "</text>"
                           "<text x=\"" val-x "\" y=\"" (+ (row-y i) 5)
                           "\" font-family=\"" svg-font-family
                           "\" font-size=\"" svg-label-fs "\" fill=\""
                           svg-text-color "\">" (svg-fmt-num (double mean-abs-value)) "</text>")))
                  shown))]
     (str "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" width "\" height=\"" h
          "\" viewBox=\"0 0 " width " " h "\" role=\"img\" aria-label=\"SHAP bar plot\">"
          "<rect x=\"0\" y=\"0\" width=\"" width "\" height=\"" h "\" fill=\"white\"/>"
          "<text x=\"" (int (/ width 2)) "\" y=\"26\" text-anchor=\"middle\" "
          "font-family=\"" svg-font-family
          "\" font-size=\"" svg-title-fs "\" font-weight=\"bold\" fill=\""
          svg-text-color "\">" (svg-escape title) "</text>"
          tick-lines
          "<line x1=\"" pad-left "\" y1=\"" pad-top
          "\" x2=\"" pad-left "\" y2=\"" (- h pad-bot)
          "\" stroke=\"" svg-axis-color "\" stroke-width=\"1\"/>"
          bars
          "<text x=\"" (+ pad-left (/ plot-w 2)) "\" y=\"" (- h 8)
          "\" text-anchor=\"middle\" font-family=\"" svg-font-family
          "\" font-size=\"" svg-axis-fs "\" fill=\""
          svg-text-color "\">mean(|SHAP value|)</text>"
          "</svg>"))))

(defn beeswarm-svg
  "Beeswarm plot of SHAP values: y = feature (ordered by mean |SHAP|),
  x = SHAP value, color = feature value (low → blue, high → red).
  Returns an `<svg>` string.

  Options:
    :top-n   number of features to show (default 25)
    :width   SVG width  (default 760)
    :height  SVG height (default auto = top-n*26 + 110)
    :title   chart title (default \"SHAP beeswarm\")
    :max-points-per-feature  cap dots per row, sampled deterministically (default 800)"
  ([explanation] (beeswarm-svg explanation {}))
  ([explanation {:keys [top-n width height title max-points-per-feature]
                 :or {top-n 25 width 640
                      title "SHAP beeswarm — per-sample SHAP value, colored by feature value"
                      max-points-per-feature 800}}]
   (let [V ^"[[D" (:values explanation)
         D ^"[[D" (:data explanation)
         feats (vec (:feature-names explanation))
         n-samples (alength V)
         ranking (expl/feature-ranking explanation)
         shown (vec (take top-n ranking))
         n-rows (count shown)
         feat->idx (into {} (map-indexed (fn [i f] [f i]) feats))
         row-h 28
         pad-top 60
         pad-bot 78
         pad-left 220
         pad-right 60
         h (long (or height (+ pad-top pad-bot (* row-h n-rows))))
         plot-w (- width pad-left pad-right)
         ;; X-range: symmetric around 0 across all shown SHAP values
         all-vals (mapcat
                    (fn [{:keys [feature]}]
                      (when-let [j (feat->idx feature)]
                        (mapv #(aget ^"[D" (aget V %) (long j)) (range n-samples))))
                    shown)
         abs-max (reduce (fn [^double m ^double v]
                           (Math/max m (Math/abs v)))
                         0.0
                         all-vals)
         abs-max (if (zero? abs-max) 1.0 abs-max)
         x-lo (- abs-max)
         x-hi abs-max
         ->x (fn [^double v]
               (+ pad-left (* (/ (- v x-lo) (- x-hi x-lo)) plot-w)))
         row-y-center (fn [^long i] (+ pad-top (* i row-h) (/ row-h 2)))
         rng (java.util.Random. 1)
         ticks (axis-ticks x-lo x-hi 7)
         tick-lines (apply str
                      (for [t ticks
                            :let [x (->x t)]]
                        (str "<line x1=\"" x "\" y1=\"" pad-top
                             "\" x2=\"" x "\" y2=\"" (- h pad-bot)
                             "\" stroke=\"" svg-grid-color "\" stroke-width=\"1\"/>"
                             "<text x=\"" x "\" y=\"" (+ (- h pad-bot) 14)
                             "\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"11\" fill=\""
                             svg-text-color "\">" (svg-fmt-num t) "</text>")))
         zero-x (->x 0.0)
         row-labels (apply str
                      (map-indexed
                        (fn [i {:keys [feature]}]
                          (str "<text x=\"" (- pad-left 8) "\" y=\""
                               (+ (row-y-center i) 4)
                               "\" text-anchor=\"end\" font-family=\"monospace\" font-size=\"11\" fill=\""
                               svg-text-color "\">"
                               (svg-escape (svg-truncate (str feature) 32))
                               "</text>"))
                        shown))
         row-bands (apply str
                     (for [i (range n-rows)
                           :when (odd? i)]
                       (str "<rect x=\"" pad-left "\" y=\""
                            (+ pad-top (* i row-h))
                            "\" width=\"" plot-w "\" height=\"" row-h
                            "\" fill=\"#fafafa\"/>")))
         dots (apply str
                (map-indexed
                  (fn [i {:keys [feature]}]
                    (let [j (long (feat->idx feature))
                          col (column-of D j)
                          [c-lo c-hi] (column-min-max col)
                          c-range (- c-hi c-lo)
                          ;; sub-sample for very large datasets
                          step (max 1 (long (Math/ceil (/ (double n-samples)
                                                          (double max-points-per-feature)))))
                          y0 (row-y-center i)
                          radius 1.8]
                      (apply str
                        (for [k (range 0 n-samples step)
                              :let [shap-v (aget ^"[D" (aget V k) j)
                                    feat-v (aget ^"[D" col k)
                                    t (if (zero? c-range) 0.5 (/ (- feat-v c-lo) c-range))
                                    color (svg-lerp-color t)
                                    cx (->x shap-v)
                                    jitter (- (* 0.7 (- (.nextDouble rng) 0.5)) 0.0)
                                    cy (+ y0 (* jitter (- row-h 6)))]]
                          (str "<circle cx=\"" (format "%.1f" cx)
                               "\" cy=\"" (format "%.1f" cy)
                               "\" r=\"" radius
                               "\" fill=\"" color
                               "\" fill-opacity=\"0.65\"/>")))))
                  shown))
         legend-w 220
         legend-x (- width pad-right legend-w)
         legend-y (- h 32)
         legend-stops (apply str
                        (for [k (range 21)
                              :let [t (/ k 20.0)
                                    x (+ legend-x (* t legend-w))
                                    color (svg-lerp-color t)]]
                          (str "<rect x=\"" (format "%.2f" x)
                               "\" y=\"" legend-y
                               "\" width=\"" (/ legend-w 20.0)
                               "\" height=\"10\" fill=\"" color "\"/>")))
         legend (str legend-stops
                     "<text x=\"" legend-x "\" y=\"" (- legend-y 4)
                     "\" font-family=\"sans-serif\" font-size=\"10\" fill=\""
                     svg-text-color "\">low</text>"
                     "<text x=\"" (+ legend-x legend-w) "\" y=\"" (- legend-y 4)
                     "\" text-anchor=\"end\" font-family=\"sans-serif\" font-size=\"10\" fill=\""
                     svg-text-color "\">high feature value</text>")]
     (str "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" width "\" height=\"" h
          "\" viewBox=\"0 0 " width " " h "\" role=\"img\" aria-label=\"SHAP beeswarm plot\">"
          "<rect x=\"0\" y=\"0\" width=\"" width "\" height=\"" h "\" fill=\"white\"/>"
          row-bands
          tick-lines
          "<line x1=\"" zero-x "\" y1=\"" pad-top
          "\" x2=\"" zero-x "\" y2=\"" (- h pad-bot)
          "\" stroke=\"#888\" stroke-dasharray=\"3 3\" stroke-width=\"1\"/>"
          "<line x1=\"" pad-left "\" y1=\"" pad-top
          "\" x2=\"" pad-left "\" y2=\"" (- h pad-bot)
          "\" stroke=\"" svg-axis-color "\" stroke-width=\"1\"/>"
          dots
          row-labels
          "<text x=\"" (int (/ width 2)) "\" y=\"22\" text-anchor=\"middle\" "
          "font-family=\"sans-serif\" font-size=\"14\" font-weight=\"bold\" fill=\""
          svg-text-color "\">" (svg-escape title) "</text>"
          "<text x=\"" (+ pad-left (/ plot-w 2)) "\" y=\"" (- h pad-bot -32)
          "\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"12\" fill=\""
          svg-text-color "\">SHAP value (impact on model output)</text>"
          legend
          "</svg>"))))

(defn waterfall-svg
  "Waterfall plot for one row: base value → top feature contributions → prediction.
  Returns an `<svg>` string.

  Options:
    :top-k    number of individual features to show; rest collapsed into
              \"+N other features\" (default 12)
    :width    SVG width (default 760)
    :height   SVG height (default auto)
    :title    chart title (default \"SHAP waterfall — row N\")"
  ([explanation row-index] (waterfall-svg explanation row-index {}))
  ([explanation row-index {:keys [top-k width height title]
                           :or {top-k 12 width 760}}]
   (let [V ^"[[D" (:values explanation)
         D ^"[[D" (:data explanation)
         i (int row-index)
         feats (vec (:feature-names explanation))
         d (count feats)
         shap-row (vec (aget V i))
         data-row (vec (aget D i))
         base (aget ^"[D" (:base-values explanation) i)
         pred (when-let [p (:prediction explanation)]
                (aget ^"[D" p i))
         ;; sum of all shap = pred - base (efficiency); compute fallback if no pred
         sum-shap (reduce + 0.0 shap-row)
         pred (if pred pred (+ base sum-shap))
         ;; rank features by |shap| desc
         ranked (->> (map (fn [f v x] {:feature f :shap (double v) :value x})
                          feats shap-row data-row)
                     (sort-by (fn [m] (- (Math/abs (double (:shap m)))))))
         shown (vec (take top-k ranked))
         rest-rows (drop top-k ranked)
         rest-sum (reduce + 0.0 (map :shap rest-rows))
         rest-row (when (seq rest-rows)
                    {:feature (str "+ " (count rest-rows) " other features")
                     :shap rest-sum :value nil})
         ;; bottom-to-top order: smallest contributions first, largest at top.
         ;; That puts the "other" bucket at the bottom.
         bars (cond->> (vec (reverse shown))
                rest-row (cons rest-row)
                true vec)
         n (count bars)
         row-h 26
         pad-top 60
         pad-bot 60
         pad-left 280
         pad-right 80
         h (long (or height (+ pad-top pad-bot (* row-h n))))
         plot-w (- width pad-left pad-right)
         ;; X-range covers the full cumulative path: base + every running prefix
         ;; sum across the (bottom-to-top) bars. Intermediate values can swing
         ;; outside [base, pred], so we expand the axis to include them.
         running-vals (reductions (fn [^double r {:keys [shap]}]
                                    (+ r (double shap)))
                                  (double base) bars)
         all-vals (concat running-vals [(double pred)])
         raw-lo (double (reduce (fn [^double a ^double b] (Math/min a b)) all-vals))
         raw-hi (double (reduce (fn [^double a ^double b] (Math/max a b)) all-vals))
         span (Math/max 1.0e-6 (- raw-hi raw-lo))
         lo (- raw-lo (* 0.10 span))
         hi (+ raw-hi (* 0.10 span))
         hi (double (Math/max hi (+ lo 0.05)))
         ->x (fn [^double v]
               (+ pad-left (* (/ (- v lo) (- hi lo)) plot-w)))
         ticks (axis-ticks lo hi 6)
         tick-lines (apply str
                      (for [t ticks
                            :let [x (->x t)]]
                        (str "<line x1=\"" x "\" y1=\"" pad-top
                             "\" x2=\"" x "\" y2=\"" (- h pad-bot)
                             "\" stroke=\"" svg-grid-color "\" stroke-width=\"1\"/>"
                             "<text x=\"" x "\" y=\"" (+ (- h pad-bot) 14)
                             "\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"11\" fill=\""
                             svg-text-color "\">" (svg-fmt-num t) "</text>")))
         row-y-top (fn [^long k] (+ pad-top (* k row-h)))
         row-y-mid (fn [^long k] (+ (row-y-top k) (/ row-h 2)))
         ;; cumulative running value, starts at base, building up to pred
         {bar-rects :rects} (reduce
                              (fn [{:keys [running rects k]} {:keys [feature shap value]}]
                                (let [start (double running)
                                      end (+ start (double shap))
                                      x0 (->x (Math/min start end))
                                      x1 (->x (Math/max start end))
                                      bar-w (Math/max 0.5 (- (double x1) (double x0)))
                                      color (if (>= (double shap) 0.0)
                                              svg-pos-color svg-neg-color)
                                      y (+ (row-y-top k) 4)
                                      bar-h (- row-h 8)
                                      label (str (svg-truncate (str feature) 38)
                                                 (when (some? value)
                                                   (str " = " (svg-fmt-num
                                                                (cond (number? value) (double value)
                                                                      :else 0.0)))))
                                      label-x (- pad-left 10)
                                      val-text (str (if (>= (double shap) 0.0) "+" "")
                                                    (svg-fmt-num (double shap)))
                                      val-x (if (>= (double shap) 0.0) (+ x1 6) (- x0 6))
                                      val-anchor (if (>= (double shap) 0.0) "start" "end")
                                      rect (str "<rect x=\"" (format "%.1f" x0)
                                                "\" y=\"" y
                                                "\" width=\"" (format "%.1f" bar-w)
                                                "\" height=\"" bar-h
                                                "\" fill=\"" color
                                                "\" fill-opacity=\"0.85\"/>"
                                                "<text x=\"" label-x "\" y=\"" (+ (row-y-mid k) 4)
                                                "\" text-anchor=\"end\" font-family=\"monospace\" font-size=\"11\" fill=\""
                                                svg-text-color "\">" (svg-escape label) "</text>"
                                                "<text x=\"" (format "%.1f" val-x)
                                                "\" y=\"" (+ (row-y-mid k) 4)
                                                "\" text-anchor=\"" val-anchor
                                                "\" font-family=\"sans-serif\" font-size=\"11\" fill=\""
                                                svg-text-color "\">" val-text "</text>")]
                                  {:running end
                                   :k (inc k)
                                   :rects (conj rects rect)}))
                              {:running base :rects [] :k 0}
                              bars)
         base-line (str "<line x1=\"" (->x base) "\" y1=\"" pad-top
                        "\" x2=\"" (->x base) "\" y2=\"" (- h pad-bot)
                        "\" stroke=\"#888\" stroke-dasharray=\"3 3\" stroke-width=\"1\"/>"
                        "<text x=\"" (->x base) "\" y=\"" (- pad-top 6)
                        "\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"11\" fill=\""
                        svg-text-color "\">E[f(X)] = " (svg-fmt-num base) "</text>")
         pred-line (str "<line x1=\"" (->x pred) "\" y1=\"" pad-top
                        "\" x2=\"" (->x pred) "\" y2=\"" (- h pad-bot)
                        "\" stroke=\"#222\" stroke-width=\"1\"/>"
                        "<text x=\"" (->x pred) "\" y=\"" (- pad-top 6)
                        "\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"11\" fill=\""
                        svg-text-color "\" font-weight=\"bold\">f(x) = " (svg-fmt-num pred) "</text>")
         title* (or title (str "SHAP waterfall — row " row-index))]
     (str "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" width "\" height=\"" h
          "\" viewBox=\"0 0 " width " " h "\" role=\"img\" aria-label=\"SHAP waterfall plot\">"
          "<rect x=\"0\" y=\"0\" width=\"" width "\" height=\"" h "\" fill=\"white\"/>"
          tick-lines
          "<line x1=\"" pad-left "\" y1=\"" pad-top
          "\" x2=\"" pad-left "\" y2=\"" (- h pad-bot)
          "\" stroke=\"" svg-axis-color "\" stroke-width=\"1\"/>"
          (apply str bar-rects)
          base-line
          pred-line
          "<text x=\"" (int (/ width 2)) "\" y=\"24\" text-anchor=\"middle\" "
          "font-family=\"sans-serif\" font-size=\"14\" font-weight=\"bold\" fill=\""
          svg-text-color "\">" (svg-escape title*) "</text>"
          "<text x=\"" (+ pad-left (/ plot-w 2)) "\" y=\"" (- h 8)
          "\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"12\" fill=\""
          svg-text-color "\">model output (P)</text>"
          "</svg>"))))
