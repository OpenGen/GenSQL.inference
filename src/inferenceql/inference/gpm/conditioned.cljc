(ns inferenceql.inference.gpm.conditioned
  (:require [inferenceql.inference.gpm.proto :as gpm.proto]))

(defrecord ConditionedGPM [gpm conditions]
  gpm.proto/GPM
  (logpdf [_ targets logpdf-conditions]
    (let [merged-conditions (merge conditions logpdf-conditions)]
      (gpm.proto/logpdf gpm targets merged-conditions)))

  (simulate [_ targets simulate-conditions]
    (let [merged-conditions (merge conditions simulate-conditions)]
      (gpm.proto/simulate gpm targets merged-conditions))))

(defn condition
  "Conditions the provided generative probabilistic model such that it only
  simulates the provided targets, and is always subject to the provided
  conditions."
  [gpm conditions]
  (assert map? conditions)
  (->ConditionedGPM gpm conditions))
