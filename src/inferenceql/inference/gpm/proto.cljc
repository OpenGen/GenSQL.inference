(ns inferenceql.inference.gpm.proto)

(defprotocol GPM
  "A simple protocol for defining a GPM."
  (logpdf             [this targets constraints])
  (simulate           [this targets constraints n-samples])
  (mutual-information [this target-a target-b constraints n-samples]))
