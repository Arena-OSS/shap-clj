(ns shap-clj.examples.tip-predictor-test
  "End-to-end validation of the tip-predictor example.  Unlike the
   per-namespace unit tests elsewhere in this project, every test here
   exercises a real (if tiny) model through the public `shap-clj.shap`
   API.  Coverage is organised by property:

     1.  Model sanity       — predictions match hand-computed numbers
     2.  Efficiency         — Σφ + base = prediction, per explainer
     3.  Cross-explainer    — kernel/permutation converge to exact
     4.  Closed-form        — linear head admits φ = w·(x−μ)
     5.  Axiom spot-checks  — dummy, symmetry, linearity on this model
     6.  Edge cases         — d=1, constant, mean-input, extremes,
                              single-row bg, batched ↔ per-row
     7.  Bookkeeping        — Explanation shape, reproducibility, plots"
  (:require [clojure.test :refer [deftest is testing]]
            [shap-clj.examples.tip-predictor :as tip]
            [shap-clj.fixtures :as fx]
            [shap-clj.plot :as plot]
            [shap-clj.shap :as shap]))

(set! *warn-on-reflection* true)

(def ^:private TOL-EXACT  1.0e-9)
(def ^:private TOL-KERNEL 1.0e-6)
(def ^:private TOL-PERM   5.0e-3)
(def ^:private TOL-LINEAR 1.0e-9)

(defn- row-at [expl i]
  (vec (aget ^"[[D" (:values expl) (int i))))

(defn- max-abs ^double [xs]
  (reduce max 0.0 (map #(Math/abs (double %)) xs)))

(defn- efficiency-max ^double [expl]
  (max-abs (shap/efficiency-residuals expl)))

(defn- column-mean ^double [rows j]
  (/ (double (reduce + (map #(double (nth % j)) rows)))
     (double (count rows))))

(defn- bg-mean [rows]
  (vec (for [j (range (count (first rows)))]
         (column-mean rows j))))

;; --- Hand-computed predictions, matched against the published weights ---
;;
;; [50  2 1] → h1 = ReLU(7.5 + 1.0)     = 8.5
;;             h2 = ReLU(5.0 − 0.6)     = 4.4
;;             y  = 8.5 + 2.2 + 0.5     = 11.20
;;
;; [25  1 0] → h1 = ReLU(3.75)          = 3.75
;;             h2 = ReLU(2.5 − 0.3)     = 2.2
;;             y  = 3.75 + 1.1 + 0.5    = 5.35
;;
;; [100 5 1] → h1 = ReLU(15 + 1)        = 16
;;             h2 = ReLU(10 − 1.5)      = 8.5
;;             y  = 16 + 4.25 + 0.5     = 20.75
;;
;; [8   4 0] → h1 = ReLU(1.2)           = 1.2
;;             h2 = ReLU(0.8 − 1.2)     = 0  ← ReLU clips
;;             y  = 1.2 + 0 + 0.5       = 1.70
(def ^:private expected-predictions [11.20 5.35 20.75 1.70])

;; ----------------------------------------------------------------------
;; 1. Model sanity
;; ----------------------------------------------------------------------

(deftest predict-matches-hand-computed-values
  (is (fx/arrays-close? expected-predictions
                        (vec (tip/predict tip/sample-orders))
                        1.0e-12)))

(deftest predict-handles-double-array-matrix
  (let [as-2d (into-array (map double-array tip/sample-orders))]
    (is (fx/arrays-close? expected-predictions
                          (vec (tip/predict as-2d))
                          1.0e-12))))

(deftest predict-row-is-consistent-with-batch
  (doseq [row tip/sample-orders]
    (is (fx/close? (tip/predict-row row)
                   (aget ^"[D" (tip/predict [row]) 0)
                   1.0e-12))))

;; ----------------------------------------------------------------------
;; 2. Efficiency: Σφ + base = prediction
;; ----------------------------------------------------------------------

(deftest efficiency-exact
  (let [e (shap/explain tip/predict tip/sample-orders
                        {:background tip/training-data :method :exact})]
    (is (< (efficiency-max e) TOL-EXACT))))

(deftest efficiency-kernel
  ;; KernelSHAP enforces efficiency by construction (one variable is
  ;; substituted out via the constraint), so the residual should be
  ;; at machine-epsilon regardless of n-samples.
  (let [e (shap/explain tip/predict tip/sample-orders
                        {:background tip/training-data :method :kernel
                         :n-samples 2000 :seed 7})]
    (is (< (efficiency-max e) 1.0e-8))))

(deftest efficiency-permutation
  (let [e (shap/explain tip/predict tip/sample-orders
                        {:background tip/training-data :method :permutation
                         :n-permutations 400 :seed 7})]
    (is (< (efficiency-max e) TOL-PERM))))

;; The linear explainer assumes an affine model; on the ReLU net its
;; attributions are the attributions of the LINEARISATION, so Σφ + base
;; equals the linear-head prediction, not f(x).  Efficiency is tested
;; for the linear-head variant further below.

;; ----------------------------------------------------------------------
;; 3. Cross-explainer agreement
;; ----------------------------------------------------------------------

(deftest kernel-matches-exact
  (let [e-exact  (shap/explain tip/predict tip/sample-orders
                               {:background tip/training-data :method :exact})
        e-kernel (shap/explain tip/predict tip/sample-orders
                               {:background tip/training-data :method :kernel
                                :n-samples 5000 :seed 11})]
    (dotimes [i (count tip/sample-orders)]
      (is (fx/arrays-close? (row-at e-exact i) (row-at e-kernel i) TOL-KERNEL)
          (str "row " i ": exact=" (row-at e-exact i)
               " kernel=" (row-at e-kernel i))))))

(deftest permutation-matches-exact
  (let [e-exact (shap/explain tip/predict tip/sample-orders
                              {:background tip/training-data :method :exact})
        e-perm  (shap/explain tip/predict tip/sample-orders
                              {:background tip/training-data :method :permutation
                               :n-permutations 2000 :seed 11})]
    (dotimes [i (count tip/sample-orders)]
      (is (fx/arrays-close? (row-at e-exact i) (row-at e-perm i) TOL-PERM)
          (str "row " i ": exact=" (row-at e-exact i)
               " perm=" (row-at e-perm i))))))

;; ----------------------------------------------------------------------
;; 4. Closed-form validation on the linear head
;;
;; When ReLU is removed the net collapses to a plain affine map
;;
;;     w = W2 · W1 = [1.0, 0.5] · [[0.15  0   1 ]
;;                                 [0.10 -0.3 0]]
;;       = [0.20, -0.15, 1.00]
;;     b = b2 = 0.5
;;
;; For any linear model with an independent masker the Shapley values
;; are φ_i = w_i · (x_i − μ_i), which we can verify by hand.
;; ----------------------------------------------------------------------

(def ^:private linear-coefs   [0.20 -0.15 1.00])
(def ^:private linear-b       0.50)
(def ^:private linear-tip-fn  (fx/linear-model linear-coefs linear-b))

(deftest linear-coefs-are-w2-dot-w1
  ;; Spelled out explicitly so the closed-form tests stay honest if
  ;; the weights ever change.
  (let [{:keys [W1 W2]} tip/weights
        d (count (first W1))
        computed (vec (for [j (range d)]
                        (reduce + (map (fn [w2k wk] (* w2k (nth wk j)))
                                       W2 W1))))]
    (is (fx/arrays-close? linear-coefs computed 1.0e-12))))

(deftest closed-form-linear-phi-matches-linear-explainer
  (let [mu (bg-mean tip/training-data)
        x  [50.0 2.0 1.0]
        expected (mapv (fn [w xi mui] (* w (- xi mui)))
                       linear-coefs x mu)
        e (shap/explain linear-tip-fn [x]
                        {:background tip/training-data :method :linear})]
    (is (fx/arrays-close? expected (row-at e 0) TOL-LINEAR))))

(deftest exact-matches-linear-explainer-on-linear-head
  (let [e-exact  (shap/explain linear-tip-fn tip/sample-orders
                               {:background tip/training-data :method :exact})
        e-linear (shap/explain linear-tip-fn tip/sample-orders
                               {:background tip/training-data :method :linear})]
    (dotimes [i (count tip/sample-orders)]
      (is (fx/arrays-close? (row-at e-exact i) (row-at e-linear i) TOL-LINEAR)))))

(deftest efficiency-linear-head-under-every-explainer
  (doseq [[label opts]
          [[:exact       {:method :exact}]
           [:kernel      {:method :kernel      :n-samples 2000 :seed 3}]
           [:permutation {:method :permutation :n-permutations 200 :seed 3}]
           [:linear      {:method :linear}]]]
    (testing (name label)
      (let [e (shap/explain linear-tip-fn tip/sample-orders
                            (merge {:background tip/training-data} opts))
            tol (case label :permutation TOL-PERM 1.0e-8)]
        (is (< (efficiency-max e) tol))))))

;; ----------------------------------------------------------------------
;; 5. Axiom spot-checks on the real (non-linear) MLP
;; ----------------------------------------------------------------------

(deftest dummy-feature-gets-zero-shap
  ;; Drop is-weekend: a model whose output is independent of feature 2.
  (let [dummy-model (fn [X]
                      (let [rows (mapv vec X)
                            n    (count rows)
                            out  (double-array n)]
                        (dotimes [i n]
                          (let [r (nth rows i)]
                            (aset out i (+ (* 0.20 (double (nth r 0)))
                                           (* -0.50 (double (nth r 1)))
                                           1.0))))
                        out))
        X [[50.0 2 1] [25.0 3 0] [80.0 1 1]]
        e (shap/explain dummy-model X
                        {:background tip/training-data :method :exact
                         :feature-names tip/feature-names})]
    (dotimes [i (count X)]
      (is (fx/close? 0.0 (nth (row-at e i) 2) 1.0e-12)
          (str "row " i " gave non-zero SHAP to dummy feature")))))

(deftest symmetric-features-get-symmetric-shap
  ;; f(x0, x1, x2) = x0 + x1 — features 0 and 1 enter identically.
  ;; Shapley symmetry requires the background to be symmetric in
  ;; the two features too; we pick a bg whose first two columns are
  ;; a pairwise permutation of each other so their joint distribution
  ;; is invariant under swapping columns 0 and 1.
  (let [sym-model (fn [X]
                    (let [rows (mapv vec X)
                          n    (count rows)
                          out  (double-array n)]
                      (dotimes [i n]
                        (let [r (nth rows i)]
                          (aset out i (+ (double (nth r 0))
                                         (double (nth r 1))))))
                      out))
        bg [[1.0  2.0  0.5]
            [2.0  1.0  0.5]
            [-1.0 -2.0 -0.5]
            [-2.0 -1.0 -0.5]]]
    (testing "equal inputs → equal SHAP"
      (let [e (shap/explain sym-model [[5.0 5.0 0.3]]
                            {:background bg :method :exact})
            r (row-at e 0)]
        (is (fx/close? (nth r 0) (nth r 1) 1.0e-12))))
    (testing "swapped inputs → swapped SHAP"
      (let [e-a (shap/explain sym-model [[5.0 3.0 0.3]]
                              {:background bg :method :exact})
            e-b (shap/explain sym-model [[3.0 5.0 0.3]]
                              {:background bg :method :exact})]
        (is (fx/close? (nth (row-at e-a 0) 0) (nth (row-at e-b 0) 1) 1.0e-12))
        (is (fx/close? (nth (row-at e-a 0) 1) (nth (row-at e-b 0) 0) 1.0e-12))))))

(deftest linearity-holds-on-tip-plus-linear
  ;; φ(α·f + β·g) = α·φ(f) + β·φ(g)
  (let [f  tip/predict
        g  (fx/linear-model [0.5 -1.0 0.3] 0.2)
        α  0.7
        β  1.2
        fg (fn [X]
             (let [yf ^"[D" (f X)
                   yg ^"[D" (g X)
                   n  (alength yf)
                   out (double-array n)]
               (dotimes [i n]
                 (aset out i (+ (* α (aget yf i))
                                (* β (aget yg i)))))
               out))
        X    [[50.0 2 1] [25.0 1 0]]
        opts {:background tip/training-data :method :exact}
        ef   (shap/explain f  X opts)
        eg   (shap/explain g  X opts)
        efg  (shap/explain fg X opts)]
    (dotimes [i (count X)]
      (let [rf  (row-at ef  i)
            rg  (row-at eg  i)
            rfg (row-at efg i)
            mix (mapv (fn [a b] (+ (* α a) (* β b))) rf rg)]
        (is (fx/arrays-close? mix rfg 1.0e-12) (str "row " i))))))

;; ----------------------------------------------------------------------
;; 6. Edge cases
;; ----------------------------------------------------------------------

(deftest single-feature-model-phi-is-fx-minus-base
  ;; d=1: φ_0 must equal f(x) − E[f(X)] exactly.
  (let [g (fn [X]
            (let [rows (mapv vec X)
                  n    (count rows)
                  out  (double-array n)]
              (dotimes [i n]
                (let [x (double (first (nth rows i)))]
                  (aset out i (+ (* 3.0 x x) x 1.0))))
              out))
        bg [[0.0] [1.0] [2.0] [3.0] [-1.0]]
        x  [[2.5]]
        base (/ (double (reduce + (vec (g bg)))) (double (count bg)))
        fx   (aget ^"[D" (g x) 0)]
    (doseq [method [:exact :kernel :permutation]]
      (testing (name method)
        (let [e (shap/explain g x (merge {:background bg :method method
                                          :n-samples 500 :n-permutations 50
                                          :seed 5}
                                         {}))]
          (is (fx/close? (- fx base) (nth (row-at e 0) 0) 1.0e-8)))))))

(deftest constant-model-returns-zero-shap-everywhere
  (let [c        42.0
        const-fn (fn [X] (double-array (count (vec X)) c))
        bg       [[1.0 2.0 3.0] [-1.0 0.5 1.5] [2.0 2.0 2.0]]
        x        [[0.1 0.2 0.3] [10.0 -5.0 7.0]]]
    (doseq [method [:exact :kernel :permutation :linear]]
      (testing (name method)
        (let [e (shap/explain const-fn x
                              (merge {:background bg :method method
                                      :n-samples 500 :n-permutations 50
                                      :seed 3}))]
          (dotimes [i (count x)]
            (is (< (max-abs (row-at e i)) 1.0e-8)
                (str "row " i " with method " method))))))))

(deftest input-at-background-mean-zero-shap-for-linear-head
  (let [mu (bg-mean tip/training-data)
        e  (shap/explain linear-tip-fn [mu]
                         {:background tip/training-data :method :exact})]
    ;; φ_i = w_i · (μ_i − μ_i) = 0
    (is (< (max-abs (row-at e 0)) 1.0e-12))))

(deftest extreme-inputs-preserve-efficiency
  (let [extreme [[1.0e6 1 0]
                 [0.0   1000 1]
                 [1.0e4 2 0]]
        e (shap/explain tip/predict extreme
                        {:background tip/training-data :method :exact})
        residual (efficiency-max e)]
    ;; Absolute tolerance is loose because of the 10^5-scale values,
    ;; but 10^-4 at those magnitudes is ~10^-10 relative.
    (is (< residual 1.0e-4))))

(deftest single-row-background-yields-valid-explanation
  (let [bg [[30.0 2 0]]
        e  (shap/explain tip/predict [[50.0 2 1]]
                         {:background bg :method :exact})
        base (aget ^"[D" (:base-values e) 0)
        pred (aget ^"[D" (:prediction e) 0)]
    ;; Efficiency still holds: the attributions explain (pred − base).
    (is (< (efficiency-max e) 1.0e-12))
    (is (fx/close? base 6.2 1.0e-12))          ; hand-computed: f([30,2,0]) = 6.2
    (is (fx/close? pred 11.2 1.0e-12))))

(deftest batched-and-per-row-explanations-agree
  (let [bg  tip/training-data
        X   tip/sample-orders
        e-batch (shap/explain tip/predict X
                              {:background bg :method :exact})
        per-row (mapv #(shap/explain tip/predict [%]
                                     {:background bg :method :exact})
                      X)]
    (dotimes [i (count X)]
      (is (fx/arrays-close? (row-at e-batch i)
                            (row-at (nth per-row i) 0)
                            1.0e-12)
          (str "row " i " differs between batched and per-row")))))

;; ----------------------------------------------------------------------
;; 7. Bookkeeping: reproducibility, shape, plots
;; ----------------------------------------------------------------------

(deftest kernel-is-reproducible-with-fixed-seed
  (let [opts {:background tip/training-data :method :kernel
              :n-samples 1500 :seed 99}
        e1 (shap/explain tip/predict tip/sample-orders opts)
        e2 (shap/explain tip/predict tip/sample-orders opts)]
    (dotimes [i (count tip/sample-orders)]
      (is (fx/arrays-close? (row-at e1 i) (row-at e2 i) 1.0e-12)))))

(deftest permutation-is-reproducible-with-fixed-seed
  (let [opts {:background tip/training-data :method :permutation
              :n-permutations 40 :seed 17}
        e1 (shap/explain tip/predict tip/sample-orders opts)
        e2 (shap/explain tip/predict tip/sample-orders opts)]
    (dotimes [i (count tip/sample-orders)]
      (is (fx/arrays-close? (row-at e1 i) (row-at e2 i) 1.0e-12)))))

(deftest explanation-has-expected-shape-and-metadata
  (let [e (shap/explain tip/predict tip/sample-orders
                        {:background tip/training-data :method :exact
                         :feature-names tip/feature-names})
        n (count tip/sample-orders)
        d (count tip/feature-names)
        V  ^"[[D" (:values e)
        D  ^"[[D" (:data e)
        bv ^"[D"  (:base-values e)
        p  ^"[D"  (:prediction e)]
    (is (= n (alength V)))
    (is (= d (alength ^"[D" (aget V 0))))
    (is (= n (alength D)))
    (is (= d (alength ^"[D" (aget D 0))))
    (is (= n (alength bv)))
    (is (= n (alength p)))
    (is (= tip/feature-names (:feature-names e)))
    (is (= :exact (:method e)))
    ;; Every row's base-value should be identical (single masker baseline).
    (dotimes [i n]
      (is (fx/close? (aget bv 0) (aget bv i) 1.0e-12)))
    ;; Prediction equals the model's batched output on the same rows.
    (let [model-pred (tip/predict tip/sample-orders)]
      (is (fx/arrays-close? (vec model-pred) (vec p) 1.0e-12)))))

(deftest plots-render-for-tip-model
  (let [e (shap/explain tip/predict tip/sample-orders
                        {:background tip/training-data :method :exact
                         :feature-names tip/feature-names})]
    (testing "bar"
      (let [p (plot/bar e)]
        (is (= 3 (count (get-in p [:data 0 :x]))))
        (is (= 3 (count (get-in p [:data 0 :y]))))))
    (testing "waterfall"
      (let [p (plot/waterfall e 0)]
        (is (= 3 (count (get-in p [:data 0 :x]))))))
    (testing "beeswarm"
      (let [p (plot/beeswarm e)]
        (is (= 3 (count (:data p))))))
    (testing "force-like"
      (let [f (plot/force-like e 0)]
        (is (= 3 (count (:contributions f))))
        (is (fx/close? (aget ^"[D" (:prediction e) 0) (:prediction f) 1.0e-12))
        ;; sum-of-shap + base = prediction (efficiency restated).
        (is (fx/close? (+ (:base-value f) (:sum-of-shap f))
                       (:prediction f)
                       1.0e-12))))))
