(ns shap-clj.linalg
  "Small linear-algebra helpers used by the KernelSHAP weighted least
  squares solve.  Operates on primitive double arrays; keeps the library
  free of native dependencies.  For production-size problems (<= few
  hundred features) this is fine; if you need faster solves plug in
  Neanderthal via shap-clj.linalg.neanderthal.")

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(defn ^"[[D" make-2d
  "Allocate a fresh m×n double matrix."
  [^long m ^long n]
  (make-array Double/TYPE m n))

(defn ^"[[D" copy-2d [^"[[D" A]
  (let [m (alength A)
        n (alength ^"[D" (aget A 0))
        B (make-2d m n)]
    (dotimes [i m]
      (System/arraycopy (aget A i) 0 (aget B i) 0 n))
    B))

(defn solve
  "Solve A x = b for square A using Gaussian elimination with
  partial pivoting.  Returns x as a double-array.  Non-destructive.
  Throws on singular A."
  ^"[D" [^"[[D" A0 ^"[D" b0]
  (let [n (alength A0)
        A (copy-2d A0)
        b (double-array n)]
    (System/arraycopy b0 0 b 0 n)
    (dotimes [k n]
      (let [pivot (loop [i k best k best-val (Math/abs (double (aget ^"[D" (aget A k) k)))]
                    (if (= i n)
                      best
                      (let [v (Math/abs (double (aget ^"[D" (aget A i) k)))]
                        (if (> v best-val)
                          (recur (inc i) i v)
                          (recur (inc i) best best-val)))))]
        (when (not= pivot k)
          (let [tmp-row (aget A k)]
            (aset A k (aget A pivot))
            (aset A pivot tmp-row))
          (let [tmp-b (aget b k)]
            (aset b k (aget b pivot))
            (aset b pivot tmp-b)))
        (let [akk (aget ^"[D" (aget A k) k)]
          (when (< (Math/abs akk) 1.0e-14)
            (throw (ex-info "Singular matrix in solve"
                            {:row k :pivot akk})))
          (loop [i (inc k)]
            (when (< i n)
              (let [factor (/ (aget ^"[D" (aget A i) k) akk)]
                (when (not (zero? factor))
                  (loop [j k]
                    (when (< j n)
                      (aset ^"[D" (aget A i) j
                            (- (aget ^"[D" (aget A i) j)
                               (* factor (aget ^"[D" (aget A k) j))))
                      (recur (inc j))))
                  (aset b i (- (aget b i) (* factor (aget b k))))))
              (recur (inc i)))))))
    ;; Back substitution.
    (let [x (double-array n)]
      (loop [i (dec n)]
        (when (>= i 0)
          (let [aii (aget ^"[D" (aget A i) i)
                sum (double
                      (loop [j (inc i) s 0.0]
                        (if (= j n)
                          s
                          (recur (inc j) (+ s (* (aget ^"[D" (aget A i) j)
                                                 (aget x j)))))))]
            (aset x i (/ (- (aget b i) sum) aii))
            (recur (dec i)))))
      x)))

(defn weighted-lstsq
  "Solve  min_x  Σ_i w_i (y_i - X_i · x)^2   for vector x.
  X is (M × D), y is length M, w is length M.  Returns x length D.
  Uses the normal equations  (Xᵀ W X) x = Xᵀ W y  with a tiny ridge."
  ^"[D" [^"[[D" X ^"[D" y ^"[D" w]
  (let [M (alength X)
        D (alength ^"[D" (aget X 0))
        XtWX (make-2d D D)
        XtWy (double-array D)]
    (when (not (and (= M (alength y)) (= M (alength w))))
      (throw (ex-info "Shape mismatch in weighted-lstsq"
                      {:rows-X M :len-y (alength y) :len-w (alength w)})))
    (dotimes [i M]
      (let [xi (aget X i)
            yi (aget y i)
            wi (aget w i)]
        (dotimes [a D]
          (let [xia (* wi (aget ^"[D" xi a))]
            (aset XtWy a (+ (aget XtWy a) (* xia yi)))
            (dotimes [b D]
              (aset ^"[D" (aget XtWX a) b
                    (+ (aget ^"[D" (aget XtWX a) b)
                       (* xia (aget ^"[D" xi b)))))))))
    (dotimes [i D]
      (aset ^"[D" (aget XtWX i) i
            (+ (aget ^"[D" (aget XtWX i) i) 1.0e-9)))
    (solve XtWX XtWy)))
