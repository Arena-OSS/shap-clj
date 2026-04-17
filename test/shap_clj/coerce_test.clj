(ns shap-clj.coerce-test
  (:require [clojure.test :refer [deftest is testing]]
            [shap-clj.coerce :as c]))

(deftest to-2d-from-seqs
  (let [X (c/to-2d [[1 2 3] [4 5 6]])]
    (is (= 2 (c/n-rows X)))
    (is (= 3 (c/n-cols X)))
    (is (= 1.0 (aget ^"[D" (c/row X 0) 0)))
    (is (= 6.0 (aget ^"[D" (c/row X 1) 2)))))

(deftest to-2d-identity-on-2d-double-array
  (let [src (c/to-2d [[1 2] [3 4]])
        same (c/to-2d src)]
    (is (identical? src same))))

(deftest to-2d-wraps-single-row
  (let [X (c/to-2d (double-array [1 2 3 4]))]
    (is (= 1 (c/n-rows X)))
    (is (= 4 (c/n-cols X)))))

(deftest to-2d-errors-on-nil
  (is (thrown? Exception (c/to-2d nil))))

(deftest to-1d-accepts-various
  (is (= 3 (count (c/to-1d [1 2 3]))))
  (is (= 3 (alength ^"[D" (c/to-1d (double-array [1 2 3]))))))

(deftest column-means
  (let [X (c/to-2d [[1 2] [3 4] [5 6]])
        mu (c/column-means X)]
    (is (= 3.0 (aget mu 0)))
    (is (= 4.0 (aget mu 1)))))
