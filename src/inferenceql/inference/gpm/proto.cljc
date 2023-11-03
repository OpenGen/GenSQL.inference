(ns inferenceql.inference.gpm.proto)

(defprotocol GPM
  "A simple protocol for defining a GPM."
  (logpdf             [this targets constraints])
  (simulate           [this targets constraints])
  (mutual-information [this target-a target-b constraints n-samples]))

(defprotocol Incorporate
  "Expand functionality of GPMs for CrossCat necessary functionality."
  (incorporate [this values] "Includes the specified values into the given GPM.")
  (unincorporate [this values] "Removes the specified values from the given GPM.
                               It is the client's responsibility to avoid unincorporating
                               values that were not previously incorporated."))

(defprotocol Score
  "Calculates the marginal log joint density of all observations and current variable configuration
  of the current state of a GPM.
  Necessary for all CrossCat-related GPMs."
  (logpdf-score [this]))

(defprotocol Insert
  "Given a non-parametric GPM and it's partition, insert a row into into the correct
  category (aka the correct table/cluster)"
  (insert [this values]))

(defprotocol Variables
  "Given a GPM, returns the variables it supports."
  (variables [this]))

(defprotocol Condition
  (condition [this conditions]))

(defprotocol Constrain
  (constrain [this event opts]))

(defprotocol LogProb
  (logprob [this event]))

(defprotocol MutualInfo
  (mutual-info [this event-a event-b]))

(defprotocol Prune
  (prune [this variables]))
