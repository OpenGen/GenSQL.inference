(ns inferenceql.inference.gpm
  (:require [inferenceql.inference.gpm.multimixture :refer [->Multimixture]]
            [inferenceql.inference.gpm.proto :as gpm-proto]))

(defn Multimixture
  "Wrapper to provide conversion to Multimixture model."
  [model]
  (->Multimixture model))

(defn logpdf
  "Given a GPM, calculates the logpdf of `targets` given `constraints`
  and optionally, `inputs`."
  ([gpm targets constraints]
   (logpdf gpm targets constraints {}))
  ([gpm targets constraints inputs]
   (gpm-proto/logpdf gpm targets constraints inputs)))

(defn mutual-information
  "Given a GPM, estimates the mutual-information of `target-a` and `target-b`
  given `constraints` with `n-samples`."
  [gpm target-a target-b constraints n-samples]
  (gpm-proto/mutual-information gpm target-a target-b constraints n-samples))

(defn simulate
  "Given a GPM, simulates `n-samples` samples of the variables in `targets`,
  given `constraints` and optionally, `inputs`."
  ([gpm targets constraints n-samples]
   (simulate gpm targets constraints n-samples {}))
  ([gpm targets constraints n-samples inputs]
   (gpm-proto/simulate gpm targets constraints n-samples inputs)))
