(ns inferenceql.inference.gpm.proto)

(defprotocol GPM
  "A simple protocol for defining a GPM."
  (logpdf             [this targets constraints])
  (simulate           [this targets constraints n-samples])
  (mutual-information [this target-a target-b constraints n-samples]))

(defprotocol Incorporate
  "Expand functionality of GPMs for CrossCat necessary functionality."
  (incorporate [this values] "Includes the specified values into the given GPM.")
  (unincorporate [this values] "Removes the specified values from the given GPM.
                               It is the client's responsibility to avoid unincorporating
                               values that were not previously incorporated."))
