(ns shap-clj.integration-test
  "End-to-end smoke tests for the user-facing API."
  (:require [clojure.test :refer [deftest is testing]]
            [shap-clj.fixtures :as fx]
            [shap-clj.plot :as plot]
            [shap-clj.shap :as shap]))

(deftest end-to-end-kernel-pipeline
  (let [f (fx/sigmoid-linear-model [2.0 -1.5 0.7 -0.4 0.1] 0.0)
        full-data (fx/det-background 200 5)
        bg (shap/background full-data 30 13)
        explainer (shap/kernel-explainer f bg {:n-samples 2000 :seed 13})
        e (explainer [[0.5 -0.5 0.3 0.1 0.2]])
        vs ^"[[D" (:values e)]
    (is (= 1 (alength vs)))
    (is (= 5 (alength ^"[D" (aget vs 0))))
    (is (< (reduce max 0.0 (map #(Math/abs (double %))
                                (shap/efficiency-residuals e)))
           1.0e-8))))

(deftest end-to-end-permutation-pipeline
  (let [f (fx/linear-model [1.0 2.0 -1.0] 0.5)
        bg (fx/det-background 50 3)
        explainer (shap/permutation-explainer f bg {:n-permutations 50 :seed 7})
        e (explainer [[0.1 0.2 0.3]])]
    (is (< (reduce max 0.0 (map #(Math/abs (double %))
                                (shap/efficiency-residuals e)))
           1.0e-3))))

(deftest plots-produce-valid-specs
  (let [f (fx/linear-model [1.0 -0.5 0.3] 0.2)
        bg (fx/det-background 20 3)
        e (shap/explain f [[0.2 -0.3 0.4] [1.0 0.5 -1.0]]
                        {:background bg :method :exact
                         :feature-names ["a" "b" "c"]})]
    (testing "bar"
      (let [p (plot/bar e)]
        (is (map? p))
        (is (= 3 (count (get-in p [:data 0 :x]))))))
    (testing "waterfall"
      (let [p (plot/waterfall e 0)]
        (is (= 3 (count (get-in p [:data 0 :x]))))))
    (testing "beeswarm"
      (let [p (plot/beeswarm e)]
        (is (= 3 (count (:data p))))))
    (testing "force-like"
      (let [p (plot/force-like e 0)]
        (is (number? (:base-value p)))
        (is (= 3 (count (:contributions p))))))))

(deftest background-sampler-is-reproducible
  (let [full (fx/det-background 100 4)
        a (shap/background full 10 42)
        b (shap/background full 10 42)]
    (is (= (alength ^"[[D" a) (alength ^"[[D" b)))
    (is (= (alength ^"[D" (aget ^"[[D" a 0))
           (alength ^"[D" (aget ^"[[D" b 0))))
    (dotimes [i (alength ^"[[D" a)]
      (dotimes [j (alength ^"[D" (aget ^"[[D" a 0))]
        (is (= (aget ^"[D" (aget ^"[[D" a i) j)
               (aget ^"[D" (aget ^"[[D" b i) j)))))))

(deftest unknown-method-errors-cleanly
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"Unknown :method"
                        (shap/explain (fx/linear-model [1.0] 0.0)
                                      [[1.0]]
                                      {:background (fx/det-background 4 1)
                                       :method :bogus}))))

(deftest missing-background-errors
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"requires :background"
                        (shap/explain (fx/linear-model [1.0] 0.0)
                                      [[1.0]] {:method :kernel}))))
