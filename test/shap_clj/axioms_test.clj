(ns shap-clj.axioms-test
  "Verifies the four defining axioms of Shapley values across every
  explainer in the library.  These are the strongest correctness
  guarantees we can assert without Python in the loop.

    1. Efficiency / local accuracy:   Σ_i φ_i + base = f(x)
    2. Symmetry:                       symmetric features share φ
    3. Dummy:                          useless features get φ = 0
    4. Linearity:                      φ of (f+g) = φ(f) + φ(g)"
  (:require [clojure.test :refer [deftest is testing]]
            [shap-clj.explainer.exact :as exact]
            [shap-clj.explainer.kernel :as kernel]
            [shap-clj.explainer.linear :as linear]
            [shap-clj.explainer.permutation :as permutation]
            [shap-clj.fixtures :as fx]
            [shap-clj.shap :as shap]))

(set! *warn-on-reflection* true)

(def ^:private TOL-EXACT 1.0e-9)
(def ^:private TOL-APPROX 0.05)

(defn- efficiency-max-abs ^double [expl]
  (let [residuals (shap/efficiency-residuals expl)]
    (reduce max 0.0 (map #(Math/abs (double %)) residuals))))

;; ---------------------------------------------------------------------------
;; 1. Efficiency
;; ---------------------------------------------------------------------------

(deftest efficiency-linear-exact
  (let [f (fx/linear-model [2.0 -1.5 0.7] 0.3)
        bg (fx/det-background 12 3)
        X [[0.1 0.5 -0.2] [1.0 -0.3 0.4]]
        e (shap/explain f X {:background bg :method :exact})]
    (is (< (efficiency-max-abs e) TOL-EXACT))))

(deftest efficiency-nonlinear-exact
  (let [f (fx/sigmoid-linear-model [1.2 -0.8 0.5] 0.1)
        bg (fx/det-background 12 3)
        X [[0.3 -0.2 0.5]]
        e (shap/explain f X {:background bg :method :exact})]
    (is (< (efficiency-max-abs e) TOL-EXACT))))

(deftest efficiency-kernel-is-satisfied
  ;; KernelSHAP is constrained to satisfy efficiency by construction
  ;; (we substitute out one variable), so the residual should be
  ;; machine-epsilon regardless of nsamples.
  (let [f (fx/sigmoid-linear-model [1.2 -0.8 0.5 0.4] 0.1)
        bg (fx/det-background 10 4)
        X [[0.3 -0.2 0.5 0.1] [-1.0 0.5 0.5 0.5]]
        e (shap/explain f X {:background bg :method :kernel :n-samples 400})]
    (is (< (efficiency-max-abs e) 1.0e-8))))

(deftest efficiency-permutation-exact-small-n
  ;; For tiny feature counts, permutation explainer with paired
  ;; permutations tends to nearly satisfy efficiency in expectation
  ;; — we just require the residual to be small.
  (let [f (fx/linear-model [2.0 -1.5] 0.3)
        bg (fx/det-background 20 2)
        X [[0.1 0.5] [1.0 -0.3]]
        e (shap/explain f X {:background bg :method :permutation :n-permutations 32})]
    (is (< (efficiency-max-abs e) TOL-APPROX))))

(deftest efficiency-linear-explainer
  (let [f (fx/linear-model [2.0 -1.5 0.7] 0.3)
        bg (fx/det-background 20 3)
        X [[0.1 0.5 -0.2]]
        e (shap/explain f X {:background bg :method :linear})]
    (is (< (efficiency-max-abs e) 1.0e-8))))

;; ---------------------------------------------------------------------------
;; 2. Symmetry
;; ---------------------------------------------------------------------------

(deftest symmetry-exact
  (let [f (fx/linear-model [1.5 1.5 0.0] 0.0)
        bg (fx/det-background 12 3)
        X [[0.7 0.7 0.5]]
        e (shap/explain f X {:background bg :method :exact})
        vs ^"[[D" (:values e)
        row ^"[D" (aget vs 0)]
    (is (fx/close? (aget row 0) (aget row 1) 1.0e-9))))

(deftest symmetry-kernel
  (let [f (fx/linear-model [1.5 1.5 0.0 0.0] 0.0)
        bg (fx/det-background 20 4)
        X [[0.7 0.7 0.5 -0.2]]
        e (shap/explain f X {:background bg :method :kernel :n-samples 2000})
        vs ^"[[D" (:values e)
        row ^"[D" (aget vs 0)]
    (is (fx/close? (aget row 0) (aget row 1) 1.0e-6))))

(deftest symmetry-permutation
  (let [f (fx/linear-model [1.5 1.5 0.0 0.0] 0.0)
        bg (fx/det-background 20 4)
        X [[0.7 0.7 0.5 -0.2]]
        e (shap/explain f X {:background bg :method :permutation :n-permutations 50})
        vs ^"[[D" (:values e)
        row ^"[D" (aget vs 0)]
    ;; Permutation is stochastic; symmetry holds in expectation. Tight bound ok for small n.
    (is (fx/close? (aget row 0) (aget row 1) TOL-APPROX))))

;; ---------------------------------------------------------------------------
;; 3. Dummy
;; ---------------------------------------------------------------------------

(deftest dummy-exact
  (let [f (fx/linear-model [2.0 0.0 -1.0] 0.3)
        bg (fx/det-background 12 3)
        X [[0.1 5.0 -0.2]]                       ; big value for dummy should not matter
        e (shap/explain f X {:background bg :method :exact})
        row ^"[D" (aget ^"[[D" (:values e) 0)]
    (is (fx/close? 0.0 (aget row 1) 1.0e-9))))

(deftest dummy-kernel
  (let [f (fx/linear-model [2.0 0.0 -1.0 0.5] 0.3)
        bg (fx/det-background 30 4)
        X [[0.1 99.0 -0.2 0.5]]
        e (shap/explain f X {:background bg :method :kernel :n-samples 4000})
        row ^"[D" (aget ^"[[D" (:values e) 0)]
    (is (< (Math/abs (aget row 1)) 1.0e-6))))

;; ---------------------------------------------------------------------------
;; 4. Linearity
;; ---------------------------------------------------------------------------

(deftest linearity-exact
  (let [f (fx/linear-model [2.0 -1.5 0.7] 0.3)
        g (fx/linear-model [-0.5 1.0 0.2] -0.1)
        ;; weights α=0.7, β=1.2
        fg (fn [X]
             (let [fy (f X) gy (g X)
                   n (alength ^"[D" fy)
                   out (double-array n)]
               (dotimes [i n]
                 (aset out i (+ (* 0.7 (aget ^"[D" fy i))
                                (* 1.2 (aget ^"[D" gy i)))))
               out))
        bg (fx/det-background 12 3)
        X [[0.2 -0.3 0.5]]
        ef  (shap/explain f X  {:background bg :method :exact})
        eg  (shap/explain g X  {:background bg :method :exact})
        efg (shap/explain fg X {:background bg :method :exact})
        rf  (vec (aget ^"[[D" (:values ef) 0))
        rg  (vec (aget ^"[[D" (:values eg) 0))
        rfg (vec (aget ^"[[D" (:values efg) 0))
        combined (mapv (fn [fi gi] (+ (* 0.7 fi) (* 1.2 gi))) rf rg)]
    (is (fx/arrays-close? combined rfg 1.0e-9))))
