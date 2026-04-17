(ns shap-clj.model-test
  (:require [clojure.test :refer [deftest is testing]]
            [shap-clj.coerce :as c]
            [shap-clj.model :as model]))

(deftest ifn-model-batched
  (let [m (model/as-model (fn [X]
                            (mapv (fn [r] (apply + r)) X)))
        out (model/predict m [[1.0 2.0] [3.0 4.0]])]
    (is (= 3.0 (aget ^"[D" out 0)))
    (is (= 7.0 (aget ^"[D" out 1)))))

(deftest ifn-model-row
  (let [m (model/as-model (fn [^"[D" r] (aget r 0)) {:row? true})
        out (model/predict m [[10.0 20.0] [30.0 40.0]])]
    (is (= 10.0 (aget ^"[D" out 0)))
    (is (= 30.0 (aget ^"[D" out 1)))))

(deftest model?-recognizes-wrapped
  (is (model/model? (model/as-model (fn [X] (mapv first X))))))

(deftest predict-coerces-inputs
  (let [m (model/as-model (fn [X] (mapv first X)))]
    (is (= 1.0 (aget ^"[D" (model/predict m [[1 2 3]]) 0)))
    (is (= 5.0 (aget ^"[D" (model/predict m (double-array [5 6 7])) 0)))))
