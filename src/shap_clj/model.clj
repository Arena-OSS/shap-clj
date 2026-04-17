(ns shap-clj.model
  "Model protocol and adapters.  A Model exposes a single batched
  prediction method `(predict model X)` where X is an (n × d) 2-D
  double array and the return value is a length-n double array.

  Any Clojure function can be used:
    - `(predict-batch X) -> double-array`  — via `(as-model f)` or direct
    - `(predict-row row)  -> double`       — via `(row-fn f)`."
  (:require [shap-clj.coerce :as c]))

(set! *warn-on-reflection* true)

(defprotocol Model
  (-predict ^"[D" [this ^"[[D" X]
    "Predict on an n×d matrix X; return a length-n double-array."))

(defn predict
  "Public entry point: predict on any tabular input.  X may be a 2-D
  double array, a seq of seqs, a Tablecloth dataset, or a single row."
  ^"[D" [model X]
  (let [X (c/to-2d X)]
    (-predict model X)))

(defrecord FnModel [f]
  Model
  (-predict [_ X]
    (c/to-1d (f X))))

(defrecord RowFnModel [f]
  Model
  (-predict [_ X]
    (let [n (c/n-rows X)
          out (double-array n)]
      (dotimes [i n]
        (aset out i (double (f (c/row X i)))))
      out)))

(defn as-model
  "Wrap an arbitrary callable as a Model.
  Options:
    :row? true  — treat f as row-at-a-time (f takes a double-array row,
                  returns a number).  Default false (batched)."
  ([f] (as-model f {}))
  ([f {:keys [row?] :or {row? false}}]
   (cond
     (satisfies? Model f) f
     row? (->RowFnModel f)
     :else (->FnModel f))))

(defn model? [x] (satisfies? Model x))
