(ns shap-clj.fixtures
  "Reusable test fixtures — small deterministic datasets and
  hand-crafted model functions whose exact Shapley values are
  known in closed form.")

(set! *warn-on-reflection* true)

(defn- dseq->2d ^"[[D" [rows]
  (let [n (count rows)
        d (count (first rows))
        out (make-array Double/TYPE n d)]
    (dotimes [i n]
      (let [r (nth rows i)
            ra ^"[D" (aget ^"[[D" out i)]
        (dotimes [j d] (aset ra j (double (nth r j))))))
    out))

(defn det-background
  "Deterministic symmetric background — columns have mean 0."
  [^long n-rows ^long d]
  (let [out (make-array Double/TYPE n-rows d)]
    (dotimes [i n-rows]
      (dotimes [j d]
        (aset ^"[D" (aget ^"[[D" out i) j
              (double (- (* 2.0 (/ (inc (mod (+ (* 31 i) (* 7 j)) n-rows))
                                   (double (inc n-rows))))
                         1.0)))))
    out))

(defn linear-model
  "Returns a fn that implements  f(x) = coefs · x + intercept,
  batched."
  [coefs intercept]
  (let [w ^"[D" (double-array coefs)
        b (double intercept)
        d (alength w)]
    (fn [X]
      (let [X (if (instance? (Class/forName "[[D") X) X (dseq->2d X))
            n (alength ^"[[D" X)
            out (double-array n)]
        (dotimes [i n]
          (let [r ^"[D" (aget ^"[[D" X i)
                s (loop [j 0 s 0.0]
                    (if (= j d)
                      s
                      (recur (inc j) (+ s (* (aget w j) (aget r j))))))]
            (aset out i (+ s b))))
        out))))

(defn interaction-model
  "f(x1, x2) = x1 * x2"
  []
  (fn [X]
    (let [X (if (instance? (Class/forName "[[D") X) X (dseq->2d X))
          n (alength ^"[[D" X)
          out (double-array n)]
      (dotimes [i n]
        (let [r ^"[D" (aget ^"[[D" X i)]
          (aset out i (* (aget r 0) (aget r 1)))))
      out)))

(defn xor-model
  "f(x1, x2, x3) where (x1 XOR x2) — features in {0,1}, x3 is dummy."
  []
  (fn [X]
    (let [X (if (instance? (Class/forName "[[D") X) X (dseq->2d X))
          n (alength ^"[[D" X)
          out (double-array n)]
      (dotimes [i n]
        (let [r ^"[D" (aget ^"[[D" X i)
              a (aget r 0) b (aget r 1)]
          (aset out i (double (if (not= (>= a 0.5) (>= b 0.5)) 1.0 0.0)))))
      out)))

(defn sigmoid-linear-model
  "f(x) = 1 / (1 + exp(-(w·x + b))) — a trivial 'DNN-like' nonlinearity."
  [coefs intercept]
  (let [w ^"[D" (double-array coefs)
        b (double intercept)
        d (alength w)]
    (fn [X]
      (let [X (if (instance? (Class/forName "[[D") X) X (dseq->2d X))
            n (alength ^"[[D" X)
            out (double-array n)]
        (dotimes [i n]
          (let [r ^"[D" (aget ^"[[D" X i)
                z (loop [j 0 s 0.0]
                    (if (= j d)
                      s
                      (recur (inc j) (+ s (* (aget w j) (aget r j))))))]
            (aset out i (/ 1.0 (+ 1.0 (Math/exp (- (+ z b))))))))
        out))))

(defn close?
  ([a b] (close? a b 1.0e-6))
  ([a b tol] (< (Math/abs (- (double a) (double b))) (double tol))))

(defn arrays-close?
  ([a b] (arrays-close? a b 1.0e-6))
  ([a b tol]
   (and (= (count a) (count b))
        (every? true? (map #(close? %1 %2 tol) a b)))))
