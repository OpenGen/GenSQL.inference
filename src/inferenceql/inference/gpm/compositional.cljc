(ns inferenceql.inference.gpm.compositional
  (:require [clojure.set :as set]
            [inferenceql.inference.dpm :as dpm]
            [inferenceql.inference.gpm.proto :as proto]
            [metaprob.distributions :as dist]))

(defn ^:private monte-carlo-integration
  [n dpm gpm targets conditions]
  (let [samples (repeatedly n #(proto/simulate gpm (dpm/independent-variables dpm) conditions))]
    (dist/logmeanexp
     (map #(proto/logpdf dpm targets %)
          samples))))

(defrecord Compositional [dpm gpm n]
  proto/Variables
  (variables [_]
    (proto/variables dpm))

  proto/GPM
  (logpdf [_ targets conditions]
    (let [target-variables (set (keys targets))
          conditioned-variables (set (keys conditions))
          dependent-variables (set (dpm/dependent-variables dpm))
          independent-variables (set (dpm/independent-variables dpm))
          dependent-targets (set/intersection target-variables dependent-variables)
          independent-targets (set/intersection target-variables independent-variables)]
      (cond (and (seq dependent-targets)
                 (seq independent-targets))
            (throw (ex-info "`inferenceql.inference.gpm/logpdf` cannot be computed for a mixture of dependent and independent target variables."
                            {:dependent-targets dependent-targets
                             :independent-targets independent-targets
                             :dpm dpm}))

            (seq independent-targets)
            (proto/logpdf gpm targets conditions)

            (seq dependent-targets)
            (if (set/subset? independent-variables conditioned-variables)
              (proto/logpdf dpm targets conditions)
              (monte-carlo-integration n dpm gpm targets conditions))))))

(defn compose
  "Composes the provided DPM and GPM in such a way that the GPM models the
  distribution on independent variables for the DPM. Conditioning for
  simulations is delegated to the GPM and the DPM. Monte Carlo integration is
  used to condition the independent variables using `n` samples."
  [n dpm gpm]
  (map->Compositional {:dpm dpm :gpm gpm :n n}))
