(ns shap-clj.adapter.deep-diamond
  "Adapter for Uncomplicate Deep Diamond trained networks.

  A Deep Diamond inference network accepts a tensor shaped
  `[batch-size n-features]` and returns a tensor shaped
  `[batch-size n-outputs]`.  We wrap it into a `shap-clj.model/Model`
  that takes a 2-D double array and returns a 1-D double array by
  taking the first output column (or a user-supplied index).

  Deep Diamond is an optional dependency: require it via the `:dd`
  alias in deps.edn.  This namespace uses `requiring-resolve` so the
  main shap-clj.* namespaces load without Deep Diamond on the classpath."
  (:require [shap-clj.coerce :as c]
            [shap-clj.model :as model]))

(set! *warn-on-reflection* true)

(defn- resolve-or-fail [sym]
  (or (requiring-resolve sym)
      (throw (ex-info "Deep Diamond is not on the classpath. Add it via the :dd alias."
                      {:symbol sym}))))

(defn- tensor-ctor [] (resolve-or-fail 'uncomplicate.diamond.tensor/tensor))
(defn- transfer!   [] (resolve-or-fail 'uncomplicate.neanderthal.core/transfer!))
(defn- infer-fn    [] (resolve-or-fail 'uncomplicate.diamond.dnn/infer))
(defn- native-T    [] (resolve-or-fail 'uncomplicate.diamond.native/diamond-factory))

(defn- tensor-desc [factory shape layout dtype]
  (let [t (tensor-ctor)]
    (t factory shape dtype layout)))

(defn dnn-model
  "Wrap a trained Deep Diamond inference network `net` as a shap-clj Model.

  Options:
    :output-index  which column of the output tensor to take (default 0)
    :layout        :nc | :nchw   (default :nc)
    :dtype         :float        (default :float)
    :factory       an explicit diamond-factory (default: default native)"
  ([net] (dnn-model net {}))
  ([net {:keys [output-index layout dtype factory]
         :or {output-index 0 layout :nc dtype :float}}]
   (reify model/Model
     (-predict [_ X]
       (let [X ^"[[D" X
             n (alength X)
             d (alength ^"[D" (aget X 0))
             factory (or factory ((native-T)))
             input-tensor ((tensor-ctor) factory [n d] dtype layout)
             ;; Flatten X into a row-major float-array for transfer!
             flat (float-array (* n d))]
         (dotimes [i n]
           (let [r ^"[D" (aget X i)]
             (dotimes [j d]
               (aset flat (+ (* i d) j) (float (aget r j))))))
         ((transfer!) flat input-tensor)
         (let [output ((infer-fn) net input-tensor)
               out-shape (try ((resolve-or-fail 'uncomplicate.diamond.tensor/shape) output)
                              (catch Throwable _ [n 1]))
               n-out (long (second out-shape))
               out-flat (float-array (* n n-out))
               _ ((transfer!) output out-flat)
               preds (double-array n)]
           (dotimes [i n]
             (aset preds i (double (aget out-flat (+ (* i n-out) (int output-index))))))
           preds))))))
