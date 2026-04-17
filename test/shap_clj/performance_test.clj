(ns shap-clj.performance-test
  "Performance budgets — assert that the library runs within
  a sensible wall-clock envelope on commodity hardware.

  The thresholds are intentionally loose (≈ 20× headroom over
  observed local timings) so CI variability doesn't flap.  If a
  change makes these fire, it's almost certainly a real regression."
  (:require [clojure.test :refer [deftest is testing]]
            [shap-clj.fixtures :as fx]
            [shap-clj.shap :as shap]))

(defmacro timing-ms [& body]
  `(let [start# (System/nanoTime)
         result# (do ~@body)
         elapsed# (/ (- (System/nanoTime) start#) 1.0e6)]
     [result# elapsed#]))

(deftest kernel-20-features-under-budget
  (let [d 20
        coefs (repeatedly d #(- (rand) 0.5))
        f (fx/sigmoid-linear-model coefs 0.0)
        bg (fx/det-background 30 d)
        X [(vec (repeat d 0.3))]
        [_ ms] (timing-ms
                (shap/explain f X {:background bg :method :kernel
                                   :n-samples 1500 :seed 1}))]
    (is (< ms 5000.0)
        (str "KernelSHAP on d=" d " took " ms " ms; expected < 5 s"))))

(deftest permutation-20-features-under-budget
  (let [d 20
        coefs (repeatedly d #(- (rand) 0.5))
        f (fx/sigmoid-linear-model coefs 0.0)
        bg (fx/det-background 30 d)
        X [(vec (repeat d 0.3))]
        [_ ms] (timing-ms
                (shap/explain f X {:background bg :method :permutation
                                   :n-permutations 20 :seed 1}))]
    (is (< ms 3000.0)
        (str "PermutationExplainer on d=" d " took " ms " ms; expected < 3 s"))))

(deftest exact-8-features-under-budget
  (let [d 8
        f (fx/linear-model (repeatedly d #(- (rand) 0.5)) 0.0)
        bg (fx/det-background 20 d)
        X [(vec (repeat d 0.3))]
        [_ ms] (timing-ms
                (shap/explain f X {:background bg :method :exact}))]
    (is (< ms 2000.0)
        (str "Exact on d=" d " took " ms " ms; expected < 2 s"))))

(deftest linear-scales-linearly-in-d
  (let [d 80
        coefs (vec (repeatedly d #(- (rand) 0.5)))
        f (fx/linear-model coefs 0.0)
        bg (fx/det-background 50 d)
        X [(vec (repeat d 0.1))]
        [_ ms] (timing-ms
                (shap/explain f X {:background bg :method :linear}))]
    (is (< ms 1500.0)
        (str "Linear on d=" d " took " ms " ms; expected < 1.5 s"))))

(deftest many-rows-kernel
  (testing "KernelSHAP on 10 rows, d=6 fits within the budget"
    (let [d 6
          f (fx/sigmoid-linear-model (repeatedly d #(- (rand) 0.5)) 0.0)
          bg (fx/det-background 30 d)
          X (vec (repeatedly 10 #(vec (repeatedly d rand))))
          [_ ms] (timing-ms
                  (shap/explain f X {:background bg :method :kernel :n-samples 800}))]
      (is (< ms 5000.0)))))
