(ns inferenceql.inference.gpm.conditioned
  (:require [inferenceql.inference.gpm.proto :as gpm.proto]))

(defrecord ConditionedGPM [gpm conditions]
  gpm.proto/GPM
  (logpdf [_ targets logpdf-conditions]
    (let [merged-conditions (merge conditions logpdf-conditions)]
      (gpm.proto/logpdf gpm targets merged-conditions)))

  (simulate [_ targets simulate-conditions]
    (let [merged-conditions (merge conditions simulate-conditions)]
      (gpm.proto/simulate gpm targets merged-conditions)))

  gpm.proto/LogProb
    (logprob [_ targets logprob-conditions]
      (let [merged-conditions (merge conditions logprob-conditions)]
       (gpm.proto/logprob gpm targets merged-conditions)))

  gpm.proto/Variables
  (variables [_]
    (gpm.proto/variables gpm))

  gpm.proto/Condition
  (condition [_ new-conditions]
    (let [merged-conditions (merge conditions new-conditions)]
      (->ConditionedGPM gpm merged-conditions))))

(defn condition
  "Conditions gpm based on conditions via rejection sampling. Arguments are the
  same as those for `inferenceql.inference.gom/condition`."
  [gpm conditions]
  (assert map? conditions)
  (->ConditionedGPM gpm conditions))
