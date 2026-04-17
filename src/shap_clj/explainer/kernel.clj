(ns shap-clj.explainer.kernel
  "KernelSHAP — the Lundberg–Lee (2017) model-agnostic estimator.

  Algorithm sketch per row x:
    1. Enumerate or sample coalitions z ∈ {0,1}^d (sizes 1..d-1)
       with Shapley-kernel weights.
    2. For each z, compute y_z = E_bg[f(x_S, X_{\\S})] via the masker.
    3. Solve the constrained weighted least squares
          min_φ  Σ_z  π(z) (y_z − φ_0 − zᵀ φ)^2
          s.t.   Σ φ = y_full − φ_0.
       We substitute out the last coordinate via the constraint to
       reduce to an unconstrained (d−1)-variable WLS."
  (:require [shap-clj.coerce :as c]
            [shap-clj.explainer.common :as common]
            [shap-clj.explanation :as expl]
            [shap-clj.linalg :as linalg]
            [shap-clj.masker :as masker]
            [shap-clj.model :as model]
            [shap-clj.shapley :as shap]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(defn- all-present ^booleans [^long n]
  (let [m (boolean-array n)]
    (dotimes [i n] (aset m i true))
    m))

(defn- all-absent ^booleans [^long n]
  (boolean-array n false))

(defn- mask->row-indicator
  "Convert a boolean coalition to a double-array of 0/1 values."
  ^"[D" [^booleans m]
  (let [n (alength m)
        out (double-array n)]
    (dotimes [i n] (aset out i (if (aget m i) 1.0 0.0)))
    out))

(defn explain-row
  "KernelSHAP attributions for one row x (double-array).
  Returns a double-array of length d."
  ^"[D" [model masker ^"[D" x {:keys [n-samples rng]
                                     :or {n-samples (+ (* 2 (alength x)) 2048)
                                          rng (java.util.Random. 42)}}]
  (let [d (alength x)]
    (if (= 1 d)
      ;; Single feature: φ_0 = f(x) - baseline.
      (let [f-full (aget (common/eval-coalitions model masker x [(all-present d)]) 0)
            f-empty (aget (common/baseline model masker) 0)
            out (double-array 1)]
        (aset out 0 (- f-full f-empty))
        out)
      (let [{:keys [coalitions weights]} (shap/build-coalitions d n-samples rng)
            M (count coalitions)
            ;; Evaluate f at the anchors (empty and full coalitions)
            anchors [(all-absent d) (all-present d)]
            anchor-preds (common/eval-coalitions model masker x anchors)
            f-empty (aget anchor-preds 0)
            f-full (aget anchor-preds 1)
            delta (- f-full f-empty)
            ;; Evaluate f at sampled coalitions
            ys (common/eval-coalitions model masker x coalitions)
            ;; Build the design matrix in reduced form.  We substitute
            ;; φ_{d-1} = delta - Σ_{i<d-1} φ_i, giving:
            ;;   y_z - φ_0 - z_{d-1} * delta ≈ Σ_{i<d-1} (z_i - z_{d-1}) φ_i
            X (linalg/make-2d M (dec d))
            eta (double-array M)]
        (dotimes [m M]
          (let [coalition ^booleans (nth coalitions m)
                last-bit (if (aget coalition (dec d)) 1.0 0.0)]
            (aset eta m (- (aget ys m) f-empty (* last-bit delta)))
            (let [row ^"[D" (aget ^"[[D" X m)]
              (dotimes [j (dec d)]
                (aset row j (- (if (aget coalition j) 1.0 0.0) last-bit))))))
        (let [phi-reduced (linalg/weighted-lstsq X eta weights)
              phi (double-array d)]
          (dotimes [i (dec d)] (aset phi i (aget phi-reduced i)))
          (let [s (double
                    (loop [i 0 s 0.0]
                      (if (= i (dec d))
                        s
                        (recur (inc i) (+ s (aget phi-reduced i))))))]
            (aset phi (dec d) (- delta s)))
          phi)))))

(defn explain
  "KernelSHAP over every row of X.  Options:
    :n-samples   int  — coalitions per row (default 2·d + 2048).
    :seed        long — RNG seed.
    :feature-names vector<string>."
  ([model background X] (explain model background X {}))
  ([model background X {:keys [n-samples seed feature-names]
                        :or {seed 42}}]
   (let [model (model/as-model model)
         masker (masker/independent background)
         X (c/to-2d X)
         n (c/n-rows X)
         d (c/n-cols X)
         rng (java.util.Random. (long seed))
         values (make-array Double/TYPE n d)
         base-val (aget (common/baseline model masker) 0)
         base-values (double-array n base-val)
         pred (model/predict model X)
         opts (cond-> {:rng rng}
                n-samples (assoc :n-samples n-samples))]
     (dotimes [i n]
       (let [phi (explain-row model masker (aget X i) opts)]
         (System/arraycopy phi 0 ^"[D" (aget ^"[[D" values i) 0 d)))
     (expl/explanation {:values values
                        :base-values base-values
                        :data X
                        :feature-names feature-names
                        :prediction pred
                        :method :kernel
                        :extra {:n-samples (or n-samples (+ (* 2 d) 2048))
                                :seed seed}}))))
