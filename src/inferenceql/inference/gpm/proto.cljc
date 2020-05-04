(ns inferenceql.inference.gpm.proto)

(defprotocol GPM
  "A simple protocol for defining a GPM."
  (logpdf             [this targets constraints]
    [this targets constraints inputs])
  (simulate           [this targets constraints n-samples]
    [this targets constraints n-samples inputs])
  (mutual-information [this target-a target-b constraints n-samples]))
