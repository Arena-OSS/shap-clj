(ns shap-clj.coerce
  "Coerce arbitrary tabular inputs (seq-of-seqs, 1-D vectors, 2-D double
  arrays, or Tablecloth datasets) into the canonical `^\"[[D\"` 2-D
  double-array representation used throughout the library.

  We deliberately avoid a hard dependency on Tablecloth here: datasets
  are detected by duck-typing so the library can be used with or
  without the Scicloj stack on the classpath.")

(set! *warn-on-reflection* true)

(defn- tablecloth-dataset?
  "Detects a scicloj/tablecloth dataset without requiring it on the classpath."
  [x]
  (let [cls (when x (.getName (class x)))]
    (and cls
         (or (.startsWith ^String cls "tech.v3.dataset")
             (.startsWith ^String cls "tablecloth")))))

(defn- dataset->2d ^"[[D" [ds]
  (let [rows (seq ds)
        d (count (first rows))
        n (count rows)
        out (make-array Double/TYPE n d)]
    (doseq [[i row] (map-indexed vector rows)]
      (let [values (if (map? row) (vec (vals row)) (vec row))
            row-arr (double-array d)]
        (dotimes [j d]
          (aset row-arr j (double (nth values j))))
        (aset ^objects out i row-arr)))
    out))

(defn ^"[[D" to-2d
  "Coerce `x` to a 2-D double array (array of double-array rows).
  Accepts a 2-D double array (returned as-is), a seq of seqs, a
  Tablecloth dataset, or a single 1-D row (wrapped to 1×d)."
  [x]
  (cond
    (nil? x)
    (throw (ex-info "to-2d received nil" {}))

    ;; Already a 2-D double array.
    (and (instance? (class (make-array Double/TYPE 0 0)) x))
    x

    (tablecloth-dataset? x)
    (dataset->2d x)

    ;; Single 1-D double-array -> wrap as one-row matrix.
    (instance? (Class/forName "[D") x)
    (let [^"[D" d x
          n (alength d)
          out (make-array Double/TYPE 1 n)]
      (System/arraycopy d 0 (aget ^objects out 0) 0 n)
      out)

    ;; A Clojure seq / vector of rows (each a seq of numbers)
    (and (sequential? x) (sequential? (first x)))
    (let [rows (vec x)
          n (count rows)
          d (count (first rows))
          out (make-array Double/TYPE n d)]
      (dotimes [i n]
        (let [r (vec (nth rows i))
              row-arr (double-array d)]
          (dotimes [j d]
            (aset row-arr j (double (nth r j))))
          (aset ^objects out i row-arr)))
      out)

    ;; A single seq of numbers -> 1×d
    (and (sequential? x) (number? (first x)))
    (let [r (vec x)
          d (count r)
          out (make-array Double/TYPE 1 d)
          row-arr (double-array d)]
      (dotimes [j d] (aset row-arr j (double (nth r j))))
      (aset ^objects out 0 row-arr)
      out)

    :else
    (throw (ex-info "Cannot coerce to 2-D double array"
                    {:type (class x)}))))

(defn ^"[D" to-1d
  "Coerce a predictions output to a 1-D double array."
  [x]
  (cond
    (nil? x)
    (throw (ex-info "to-1d received nil" {}))

    (instance? (Class/forName "[D") x)
    x

    (and (sequential? x) (every? number? x))
    (double-array x)

    ;; Some frameworks return a nested 1-column output, flatten.
    (and (sequential? x) (sequential? (first x)))
    (double-array (map (comp double first) x))

    :else
    (throw (ex-info "Cannot coerce to 1-D double array"
                    {:type (class x)}))))

(defn n-rows ^long [^"[[D" X] (alength X))
(defn n-cols ^long [^"[[D" X] (alength ^"[D" (aget X 0)))

(defn row ^"[D" [^"[[D" X ^long i] (aget X i))

(defn ^"[[D" rows->2d
  "Build a 2-D double array from a collection of double-arrays."
  [rows]
  (let [n (count rows)
        d (alength ^"[D" (first rows))
        out (make-array Double/TYPE n d)]
    (dotimes [i n]
      (System/arraycopy ^"[D" (nth rows i) 0 (aget ^objects out i) 0 d))
    out))

(defn column-means ^"[D" [^"[[D" X]
  (let [n (alength X)
        d (alength ^"[D" (aget X 0))
        means (double-array d)]
    (dotimes [i n]
      (let [r (aget X i)]
        (dotimes [j d]
          (aset means j (+ (aget means j) (aget ^"[D" r j))))))
    (dotimes [j d]
      (aset means j (/ (aget means j) (double n))))
    means))
