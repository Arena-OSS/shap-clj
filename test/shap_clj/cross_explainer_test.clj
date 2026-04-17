(ns shap-clj.cross-explainer-test
  "Cross-check: on the same small problem, every explainer should
  agree with the Exact explainer to within its expected tolerance."
  (:require [clojure.test :refer [deftest is testing]]
            [shap-clj.fixtures :as fx]
            [shap-clj.shap :as shap]))

(defn- diffs ^double [expected actual]
  (let [ev (vec (aget ^"[[D" (:values expected) 0))
        av (vec (aget ^"[[D" (:values actual) 0))]
    (reduce max 0.0 (map (fn [e a] (Math/abs (- (double e) (double a)))) ev av))))

(deftest kernel-matches-exact-on-sigmoid-linear
  (let [f (fx/sigmoid-linear-model [1.2 -0.8 0.5 0.4 -0.2] 0.1)
        bg (fx/det-background 10 5)
        X [[0.3 -0.2 0.5 0.1 -0.4]]
        e-exact  (shap/explain f X {:background bg :method :exact})
        e-kernel (shap/explain f X {:background bg :method :kernel :n-samples 4000 :seed 13})]
    (is (< (diffs e-exact e-kernel) 1.0e-6))))

(deftest permutation-matches-exact-on-linear
  (let [f (fx/linear-model [1.2 -0.8 0.5 0.4] 0.1)
        bg (fx/det-background 20 4)
        X [[0.3 -0.2 0.5 0.1]]
        e-exact (shap/explain f X {:background bg :method :exact})
        e-perm  (shap/explain f X {:background bg :method :permutation :n-permutations 200 :seed 13})]
    ;; On a linear model perm converges quickly; allow small tolerance.
    (is (< (diffs e-exact e-perm) 1.0e-3))))

(deftest linear-matches-exact-on-linear-model
  (let [f (fx/linear-model [1.2 -0.8 0.5] 0.1)
        bg (fx/det-background 15 3)
        X [[0.3 -0.2 0.5] [-1.0 2.0 0.0]]
        e-exact  (shap/explain f X {:background bg :method :exact})
        e-linear (shap/explain f X {:background bg :method :linear})]
    (dotimes [i 2]
      (let [re (vec (aget ^"[[D" (:values e-exact) i))
            rl (vec (aget ^"[[D" (:values e-linear) i))]
        (is (fx/arrays-close? re rl 1.0e-9)
            (str "row " i " exact=" re " linear=" rl))))))

(deftest kernel-equals-permutation-on-interaction
  ;; With true interactions both approximate explainers should
  ;; still agree with each other closely enough, in expectation.
  (let [f (fx/interaction-model)                 ; f(x1,x2) = x1*x2 (2 features)
        bg (fx/det-background 20 2)
        X [[1.0 2.0]]
        e-kern (shap/explain f X {:background bg :method :kernel :n-samples 2000 :seed 13})
        e-perm (shap/explain f X {:background bg :method :permutation :n-permutations 200 :seed 13})
        rk (vec (aget ^"[[D" (:values e-kern) 0))
        rp (vec (aget ^"[[D" (:values e-perm) 0))]
    (is (fx/arrays-close? rk rp 1.0e-2))))
