(ns shap-clj.explainer.linear
  "LinearExplainer (independent features).

  For a linear model  f(x) = w · x + b  the Shapley values under
  feature-independence are
       φ_i = w_i · (x_i − μ_i)
  where μ is the mean of the background distribution, and the
  base value is  w · μ + b.

  The coefficients can either be supplied explicitly via `:coefs` and
  `:intercept`, or inferred by probing the model on the background
  (which works for any affine function, even if only a callable is
  given)."
  (:require [shap-clj.coerce :as c]
            [shap-clj.explanation :as expl]
            [shap-clj.model :as model]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(defn infer-coefs
  "Given a model and a background dataset, recover (w, b) assuming
  the model is affine.  Uses finite differences: f(μ + e_i) − f(μ)
  gives w_i, and f(μ) gives w·μ + b."
  [model background]
  (let [bg (c/to-2d background)
        model (model/as-model model)
        ^"[D" mu (c/column-means bg)
        d (alength mu)
        mu-batch ^"[[D" (make-array Double/TYPE 1 d)
        _ (aset mu-batch 0 mu)
        f-mu (double (aget ^"[D" (model/predict model mu-batch) 0))
        batch ^"[[D" (make-array Double/TYPE d d)
        _ (dotimes [i d]
            (let [row ^"[D" (aget batch i)]
              (System/arraycopy mu 0 row 0 d)
              (aset row i (+ (aget row i) 1.0))))
        preds ^"[D" (model/predict model batch)
        w (double-array d)]
    (dotimes [i d]
      (aset w i (- (aget preds i) f-mu)))
    ;; f-mu = w·μ + b  =>  b = f-mu - w·μ
    (let [wmu (double
                (loop [i 0 s 0.0]
                  (if (= i d) s
                      (recur (inc i) (+ s (* (aget w i) (aget mu i)))))))]
      {:coefs w
       :intercept (- f-mu wmu)
       :mean mu})))

(defn explain
  "LinearExplainer (independent features) over every row of X.
  Options:
    :coefs      double-array of length d (explicit coefficients)
    :intercept  double (explicit intercept)
    :feature-names

  If :coefs is omitted, coefs are inferred from the model via
  finite differences on the background."
  ([model background X] (explain model background X {}))
  ([model background X {:keys [coefs intercept feature-names]}]
   (let [model (model/as-model model)
         bg (c/to-2d background)
         X (c/to-2d X)
         n (c/n-rows X)
         d (c/n-cols X)
         {inferred-w :coefs inferred-b :intercept mu :mean}
         (if (and coefs intercept)
           {:coefs (c/to-1d coefs) :intercept (double intercept)
            :mean (c/column-means bg)}
           (infer-coefs model bg))
         w ^"[D" inferred-w
         b (double inferred-b)
         values (make-array Double/TYPE n d)
         base-val (let [s (double
                            (loop [i 0 s 0.0]
                              (if (= i d) s
                                  (recur (inc i) (+ s (* (aget w i) (aget ^"[D" mu i)))))))]
                    (+ s b))
         base-values (double-array n base-val)
         pred (model/predict model X)]
     (dotimes [i n]
       (let [row ^"[D" (aget X i)
             out-row ^"[D" (aget ^"[[D" values i)]
         (dotimes [j d]
           (aset out-row j (* (aget w j) (- (aget row j) (aget ^"[D" mu j)))))))
     (expl/explanation {:values values
                        :base-values base-values
                        :data X
                        :feature-names feature-names
                        :prediction pred
                        :method :linear
                        :extra {:coefs (vec w)
                                :intercept b
                                :mean (vec mu)}}))))
