(ns shap-clj.linalg-test
  (:require [clojure.test :refer [deftest is testing]]
            [shap-clj.fixtures :as fx]
            [shap-clj.linalg :as linalg]))

(defn- ^"[[D" matrix [& rows]
  (let [n (count rows)
        d (count (first rows))
        out (make-array Double/TYPE n d)]
    (dotimes [i n]
      (let [r (vec (nth rows i))]
        (dotimes [j d]
          (aset ^"[D" (aget ^"[[D" out i) j (double (nth r j))))))
    out))

(deftest solve-diagonal-system
  (let [A (matrix [2.0 0.0] [0.0 3.0])
        b (double-array [4.0 9.0])
        x (linalg/solve A b)]
    (is (fx/close? 2.0 (aget x 0)))
    (is (fx/close? 3.0 (aget x 1)))))

(deftest solve-dense-system
  (let [A (matrix [2.0 1.0 -1.0]
                  [-3.0 -1.0 2.0]
                  [-2.0 1.0 2.0])
        b (double-array [8.0 -11.0 -3.0])
        x (linalg/solve A b)]
    (is (fx/arrays-close? [2.0 3.0 -1.0] (vec x) 1.0e-9))))

(deftest weighted-lstsq-recovers-linear-coefs
  ;; y = 1.5 x1 - 0.5 x2 + 2.0
  ;; we'll augment with a constant column and recover [2.0 1.5 -0.5]
  (let [X (matrix [1.0 1.0 2.0]
                  [1.0 2.0 1.0]
                  [1.0 0.5 -1.0]
                  [1.0 3.0 0.0]
                  [1.0 -2.0 -2.0])
        true-coefs [2.0 1.5 -0.5]
        y (double-array (mapv (fn [r] (+ (nth true-coefs 0)
                                         (* (nth true-coefs 1) (nth r 1))
                                         (* (nth true-coefs 2) (nth r 2))))
                              (map vec [[1.0 1.0 2.0]
                                        [1.0 2.0 1.0]
                                        [1.0 0.5 -1.0]
                                        [1.0 3.0 0.0]
                                        [1.0 -2.0 -2.0]])))
        w (double-array (repeat 5 1.0))
        x (linalg/weighted-lstsq X y w)]
    (is (fx/arrays-close? true-coefs (vec x) 1.0e-6))))

(deftest weighted-lstsq-respects-weights
  ;; Big-weight row pins the solution to pass through it.
  (let [X (matrix [1.0 0.0]
                  [1.0 1.0]
                  [1.0 2.0])
        y (double-array [10.0 0.0 0.0])
        w (double-array [1000.0 1.0 1.0])
        x (linalg/weighted-lstsq X y w)
        pred-0 (+ (aget x 0) (* 0.0 (aget x 1)))]
    (is (< (Math/abs (- pred-0 10.0)) 0.1))))
