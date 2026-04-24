(ns shap-clj.examples.tip-predictor
  "End-to-end example: a tiny 2-layer ReLU neural network that predicts
   a dinner tip (in dollars) from three features:

     bill        - subtotal before tip, in dollars
     party-size  - number of diners at the table
     is-weekend  - 0.0 / 1.0 indicator

   Architecture:

       h1 = ReLU(0.15·bill + 0.00·party-size + 1.00·is-weekend)
       h2 = ReLU(0.10·bill − 0.30·party-size + 0.00·is-weekend)
       y  = 1.0·h1 + 0.5·h2 + 0.5

   All weights are hand-chosen rather than trained, so every prediction
   and every SHAP attribution is fully deterministic.  The example is
   small enough that:

     * predictions can be hand-verified
     * the exact Shapley explainer (O(2^d)) runs instantly
     * the linearisation (w = W2·W1) admits the closed-form
       φ_i = w_i (x_i − μ_i) under an independent masker

   which is everything we need to validate SHAP end-to-end.

   Run the demo with:

       clj -M:example"
  (:require [shap-clj.shap :as shap]))

(set! *warn-on-reflection* true)

(def feature-names ["bill" "party-size" "is-weekend"])

(def weights
  "Network parameters as plain data so experiments can tweak them
   without touching the prediction code.  W1 is (hidden × input);
   W2 is (output × hidden)."
  {:W1 [[0.15  0.00 1.00]
        [0.10 -0.30 0.00]]
   :b1 [0.0 0.0]
   :W2 [1.0 0.5]
   :b2 0.5})

(defn- relu ^double [^double x]
  (if (pos? x) x 0.0))

(defn predict-row
  "Predict the tip for a single row (any indexable of length 3)."
  ^double [x]
  (let [bill    (double (nth x 0))
        party   (double (nth x 1))
        weekend (double (nth x 2))
        h1 (relu (+ (* 0.15 bill) (* 0.00 party) (* 1.00 weekend)))
        h2 (relu (+ (* 0.10 bill) (* -0.30 party) (* 0.00 weekend)))]
    (+ (* 1.0 h1) (* 0.5 h2) 0.5)))

(def ^:private double-array-2d-class (Class/forName "[[D"))

(defn predict
  "Batched prediction.  Accepts either a 2-D double array or a seq of
   seqs; returns a length-n double-array of tip predictions."
  ^"[D" [X]
  (if (instance? double-array-2d-class X)
    (let [X ^"[[D" X
          n (alength X)
          out (double-array n)]
      (dotimes [i n]
        (aset out i (predict-row (aget X i))))
      out)
    (let [rows (vec X)
          n (count rows)
          out (double-array n)]
      (dotimes [i n]
        (aset out i (predict-row (nth rows i))))
      out)))

(def training-data
  "A deterministic toy history of dinner checks, used as the SHAP
   background distribution.  Twelve rows is small enough for exact
   enumeration during testing and large enough that the mean is
   non-degenerate."
  [[20.0 1 0]
   [35.0 2 0]
   [50.0 2 1]
   [15.0 1 0]
   [45.0 3 1]
   [75.0 4 1]
   [25.0 1 0]
   [60.0 3 0]
   [90.0 5 1]
   [40.0 2 1]
   [30.0 2 0]
   [55.0 3 1]])

(def sample-orders
  "Four representative orders we'd like explained.  The last one
   triggers ReLU clipping on the second hidden unit."
  [[ 50.0 2 1]
   [ 25.0 1 0]
   [100.0 5 1]
   [  8.0 4 0]])

(defn- fmt-row [row]
  (format "bill=%6.2f party=%d weekend=%d"
          (double (nth row 0))
          (long   (nth row 1))
          (long   (nth row 2))))

(defn -main
  "Print predictions and SHAP attributions for every row of
   `sample-orders`, under every built-in explainer."
  [& _]
  (let [X      sample-orders
        bg     training-data
        preds  (predict X)
        n      (count X)]
    (println "=== shap-clj example: tip predictor ===")
    (println)
    (println "Features      :" feature-names)
    (println "Background rows:" (count bg))
    (println)
    (println "--- Predictions ---")
    (dotimes [i n]
      (println (format "Order %d  %s  →  tip $%.2f"
                       i (fmt-row (nth X i)) (aget ^"[D" preds i))))
    (println)
    (doseq [[label opts]
            [[:exact       {:method :exact}]
             [:kernel      {:method :kernel       :n-samples 4000 :seed 0}]
             [:permutation {:method :permutation  :n-permutations 400 :seed 0}]
             [:linear      {:method :linear}]]]
      (println (format "--- SHAP (%s) ---" (name label)))
      (let [e (shap/explain predict X
                            (merge {:background bg :feature-names feature-names}
                                   opts))
            bv (aget ^"[D" (:base-values e) 0)]
        (println (format "base value (E[f(X)]): %.4f" bv))
        (dotimes [i n]
          (println (format "Order %d (pred %.2f)" i (aget ^"[D" preds i)))
          (let [row (aget ^"[[D" (:values e) i)]
            (dotimes [j (count feature-names)]
              (println (format "  %-10s φ = %+7.4f"
                               (nth feature-names j)
                               (aget ^"[D" row j)))))))
      (println))))
