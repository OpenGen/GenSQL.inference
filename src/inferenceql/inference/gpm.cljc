(ns inferenceql.inference.gpm
  (:require [inferenceql.inference.gpm.multimixture :refer [->Multimixture]]
            [inferenceql.inference.gpm.proto :as gpm-proto]))

(defn Multimixture
  "Wrapper to provide conversion to Multimixture model."
  [model]
  (->Multimixture model))

(defn gpm?
  "Returns `true` if `x` is a generative probabilistic model."
  [x]
  (satisfies? gpm-proto/GPM x))

(defn logpdf
  "Given a GPM, calculates the logpdf of `targets` given `constraints`."
  [gpm targets constraints]
  (gpm-proto/logpdf gpm targets constraints))

(defn mutual-information
  "Given a GPM, estimates the mutual-information of `target-a` and `target-b`
  given `constraints` with `n-samples`."
  [gpm target-a target-b constraints n-samples]
  (gpm-proto/mutual-information gpm target-a target-b constraints n-samples))

(defn simulate
  "Given a GPM, simulates `n-samples` samples of the variables in `targets`,
  given `constraints`."
  [gpm targets constraints n-samples]
  (gpm-proto/simulate gpm targets constraints n-samples))
