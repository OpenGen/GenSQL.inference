(ns inferenceql.inference.gpm
  (:require [inferenceql.inference.gpm.multimixture :refer [->Multimixture]]
            #?(:clj [inferenceql.inference.gpm.http :as http])
            [inferenceql.inference.gpm.proto :as gpm-proto]))

#?(:clj
   (defn http
     "Returns a generative probabilistic model (GPM) that computes results by
  forwarding requests over HTTP. `url` should include the URL scheme and
  authority. Functions that operate on GPMs will trigger HTTP requests to an
  endpoint at `url` with function name as the path. For example, if `url` is
  `http://localhost`, calling `logpdf` will trigger a request to
  `http://localhost/logpdf`. Arguments will be included as JSON in the request
  body as a single JSON object with the argument names being used as keys.
  Responses will be read as JSON with the keys being coerced to keywords."
     [url]
     (http/->HTTP url)))

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
