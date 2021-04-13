(ns inferenceql.inference.gpm.conditioned
  (:require [clojure.set :as set]
            [inferenceql.inference.gpm.proto :as gpm.proto]))

(defrecord ConditionedGPM [gpm targets conditions]
  gpm.proto/GPM
  (logpdf [_ logpdf-targets logpdf-conditions]
    (let [merged-targets (select-keys logpdf-targets targets)
          merged-conditions (merge conditions logpdf-conditions)]
      (gpm.proto/logpdf gpm merged-targets merged-conditions)))

  (simulate [_ simulate-targets simulate-conditions]
    (let [merged-targets (set/intersection (set targets) (set simulate-targets))
          merged-conditions (merge conditions simulate-conditions)]
      (gpm.proto/simulate gpm merged-targets merged-conditions)))

  gpm.proto/Variables
  (variables [_]
    (set/intersection targets (gpm.proto/variables gpm))))

(defn condition
  "Conditions the provided generative probabilistic model such that it only
  simulates the provided targets, and is always subject to the provided
  conditions."
  [gpm targets conditions]
  (assert vector? targets)
  (assert map? conditions)
  (->ConditionedGPM gpm targets conditions))
