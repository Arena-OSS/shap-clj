(ns shap-clj.explanation-test
  (:require [clojure.test :refer [deftest is]]
            [shap-clj.explanation :as expl]
            [shap-clj.fixtures :as fx]
            [shap-clj.shap :as shap]))

(deftest explanation-metadata
  (let [f (fx/linear-model [2.0 -1.0] 0.5)
        bg (fx/det-background 10 2)
        e (shap/explain f [[1.0 2.0] [-1.0 0.5]] {:background bg :method :exact
                                                  :feature-names ["alpha" "beta"]})]
    (is (= :exact (:method e)))
    (is (= ["alpha" "beta"] (:feature-names e)))
    (is (= 2 (alength ^"[[D" (:values e))))
    (is (= 2 (alength ^"[D" (aget ^"[[D" (:values e) 0))))
    (is (some? (:prediction e)))))

(deftest default-feature-names
  (let [f (fx/linear-model [1.0 2.0 3.0] 0.0)
        bg (fx/det-background 10 3)
        e (shap/explain f [[0.1 0.1 0.1]] {:background bg :method :exact})]
    (is (= ["f0" "f1" "f2"] (:feature-names e)))))

(deftest mean-abs-values-ordering
  (let [f (fx/linear-model [5.0 0.0 1.0] 0.0)
        bg (fx/det-background 10 3)
        e (shap/explain f [[1.0 10.0 -1.0]] {:background bg :method :exact})
        ranking (expl/feature-ranking e)]
    ;; Dummy feature (index 1) should end up last.
    (is (= "f1" (:feature (last ranking))))
    ;; Feature 0 has the biggest coefficient × sizeable residual → top.
    (is (= "f0" (:feature (first ranking))))))

(deftest row-extraction
  (let [f (fx/linear-model [2.0 -1.0] 0.0)
        bg (fx/det-background 10 2)
        e  (shap/explain f [[1.0 2.0] [-3.0 4.0]] {:background bg :method :exact})
        r0 (expl/row e 0)]
    (is (= 1 (alength ^"[[D" (:values r0))))))
