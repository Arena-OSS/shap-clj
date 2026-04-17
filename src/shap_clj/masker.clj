(ns shap-clj.masker
  "Maskers turn `(x, coalition-bitmap)` into a batch of synthetic rows
  that represent `E[f(X) | X_S = x_S]` under some simplifying assumption.

  The independent masker is what KernelSHAP / PermutationExplainer
  expect for tabular models: it draws rows from a `background` data
  set and replaces the positions where the coalition is `true` with
  the corresponding values from `x`."
  (:require [shap-clj.coerce :as c]))

(set! *warn-on-reflection* true)

(defprotocol Masker
  (-mask [this ^"[D" x ^booleans coalition]
    "Return an (B × d) batch of rows representing the masking of x
    under `coalition`.  B is implementation-defined (typically the
    size of the background data)."))

(deftype IndependentMasker [^"[[D" bg]
  Masker
  (-mask [_ x coalition]
    (let [bg* ^"[[D" bg
          ^"[D" x* x
          ^booleans coalition* coalition
          B (alength bg*)
          d (alength x*)
          out (make-array Double/TYPE B d)]
      (dotimes [b B]
        (let [bg-row ^"[D" (aget bg* b)
              out-row ^"[D" (aget ^"[[D" out b)]
          (dotimes [j d]
            (aset out-row j
                  (double (if (aget coalition* j)
                            (aget x* j)
                            (aget bg-row j)))))))
      out)))

(defn independent
  "Create an IndependentMasker from a background dataset.
  The background should be a representative sample (say 50–500 rows)
  of the training distribution.  Larger backgrounds give more
  accurate conditional expectations but cost more to evaluate."
  [background]
  (->IndependentMasker (c/to-2d background)))

(defn masker? [x] (satisfies? Masker x))

(defn ^"[[D" mask
  "Apply a masker to (x, coalition)."
  [masker ^"[D" x ^booleans coalition]
  (-mask masker x coalition))

(defn background-size ^long [^IndependentMasker m] (alength ^"[[D" (.-bg m)))

(defn background-data ^"[[D" [^IndependentMasker m] (.-bg m))
