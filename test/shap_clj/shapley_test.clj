(ns shap-clj.shapley-test
  (:require [clojure.test :refer [deftest is testing]]
            [shap-clj.shapley :as shap]))

(deftest binomial-test
  (is (= 1.0 (shap/binomial 0 0)))
  (is (= 10.0 (shap/binomial 5 2)))
  (is (= 252.0 (shap/binomial 10 5)))
  (is (= 1.0 (shap/binomial 10 10)))
  (is (= 0.0 (shap/binomial 5 6)))
  (is (= 0.0 (shap/binomial 5 -1))))

(deftest shapley-kernel-weights-sum-to-what-they-should
  (testing "enumerating all coalitions of each size reproduces the kernel sum"
    (doseq [n (range 2 8)]
      (let [per-size (mapv #(* (shap/binomial n %)
                               (shap/shapley-kernel-weight n %))
                           (range 1 n))
            total (reduce + per-size)]
        ;; Σ_{s=1}^{n-1} C(n,s) * π(s) = Σ_{s=1}^{n-1} (n-1)/(s(n-s))
        ;;                             = 2(n-1) * Σ_{s=1}^{⌊n/2⌋} 1/(s(n-s))     (paired)
        ;; We verify the equivalence directly.
        (let [direct (reduce + (for [s (range 1 n)]
                                 (/ (double (dec n))
                                    (* s (- n s)))))]
          (is (< (Math/abs (- total direct)) 1.0e-9)
              (str "n=" n " total=" total " direct=" direct)))))))

(deftest enumerate-coalitions-is-exhaustive
  (let [{:keys [coalitions]} (shap/enumerate-coalitions 5)]
    (is (= 30 (count coalitions)))        ; 2^5 - 2 = 30
    (doseq [s (range 1 5)]
      (is (= (shap/binomial 5 s)
             (double (count (filter #(= s (shap/mask-size %)) coalitions))))))))

(deftest mask-helpers
  (let [m (shap/indices->mask 4 [0 2])]
    (is (= 2 (shap/mask-size m)))
    (is (= [0 2] (shap/mask->indices m)))
    (let [c (shap/mask-complement m)]
      (is (= [1 3] (shap/mask->indices c)))
      (is (= 4 (+ (shap/mask-size m) (shap/mask-size c)))))))

(deftest sampled-coalitions-honor-kernel-on-average
  (testing "Paired sampling yields ~uniform-kernel-weighted size distribution."
    (let [n 6
          budget 20000
          rng (java.util.Random. 7)
          {:keys [sizes]} (shap/sample-coalitions n budget rng)
          freqs (frequencies (seq sizes))
          empirical (into {} (map (fn [[k v]] [k (/ v (double budget))])) freqs)
          theoretical (let [w (for [s (range 1 n)]
                                [s (shap/size-kernel-weight n s)])
                            z (reduce + (map second w))]
                        (into {} (map (fn [[s v]] [s (/ v z)])) w))]
      (doseq [s (range 1 n)]
        (is (< (Math/abs (- (get empirical s 0.0)
                            (get theoretical s 0.0)))
               0.025)
            (str "size " s " empirical " (get empirical s) " theoretical " (get theoretical s))))))
  )
