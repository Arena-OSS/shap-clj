(ns shap-clj.explainer.common
  "Shared helpers for explainers: running a model across a stacked
  (M × B × d) batch of masked rows and averaging per coalition."
  (:require [shap-clj.coerce :as c]
            [shap-clj.masker :as masker]
            [shap-clj.model :as model]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(defn ^"[D" eval-coalitions
  "Given a list of coalitions (boolean-arrays) and instance x, build
  one large (M*B × d) batch of masked rows, call the model once, then
  average the resulting predictions across the B background rows for
  each coalition.  Returns a length-M double-array of conditional
  expectations E[f | X_S = x_S] estimated by the masker."
  [model masker ^"[D" x coalitions]
  (let [M (count coalitions)
        bg ^"[[D" (masker/background-data masker)
        B (alength bg)
        d (alength x)
        batch (make-array Double/TYPE (* M B) d)]
    (dotimes [m M]
      (let [coalition ^booleans (nth coalitions m)]
        (dotimes [b B]
          (let [bg-row ^"[D" (aget bg b)
                out-row ^"[D" (aget ^"[[D" batch (+ (* m B) b))]
            (dotimes [j d]
              (aset out-row j
                    (if (aget coalition j)
                      (aget x j)
                      (aget bg-row j))))))))
    (let [preds ^"[D" (model/predict model batch)
          out (double-array M)]
      (dotimes [m M]
        (let [base (* m B)
              s (double
                  (loop [b 0 s 0.0]
                    (if (= b B)
                      s
                      (recur (inc b) (+ s (aget preds (+ base b)))))))]
          (aset out m (/ s (double B)))))
      out)))

(defn ^"[D" baseline
  "The masker-level expected prediction: f applied to the background
  and averaged.  This becomes E[f(X)] for an independent masker."
  [model masker]
  (let [bg ^"[[D" (masker/background-data masker)
        preds ^"[D" (model/predict model bg)
        n (alength preds)
        s (double
            (loop [i 0 s 0.0]
              (if (= i n) s (recur (inc i) (+ s (aget preds i))))))]
    (let [a (double-array 1)]
      (aset a 0 (/ s (double n)))
      a)))
