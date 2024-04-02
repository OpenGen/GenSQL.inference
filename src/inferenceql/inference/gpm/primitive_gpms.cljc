(ns inferenceql.inference.gpm.primitive-gpms
  (:require [inferenceql.inference.gpm.primitive-gpms.bernoulli :as bernoulli]
            [inferenceql.inference.gpm.primitive-gpms.categorical :as categorical]
            [inferenceql.inference.gpm.primitive-gpms.gaussian :as gaussian]
            [inferenceql.inference.gpm.primitive-gpms.poisson :as poisson]))

(defn primitive?
  "Checks whether the given GPM is a primitive GPM."
  [stattype]
  (and (record? stattype)
       (or (bernoulli/bernoulli? stattype)
           (categorical/categorical? stattype)
           (gaussian/gaussian? stattype)
           (poisson/poisson? stattype))))

(defn hyper-grid
  [stattype data & {:keys [n-grid] :or {n-grid 30}}]
  (if (empty? data)
    {}
    (case stattype
      :bernoulli (bernoulli/hyper-grid data n-grid)
      :categorical (categorical/hyper-grid data n-grid)
      :gaussian (gaussian/hyper-grid data n-grid)
      :poisson (poisson/hyper-grid data n-grid)
      (throw (ex-info (str "pGPM doesn't exist: " stattype)
                      {:stattype stattype :data data})))))

(defn export-category
  [stattype category]
  (case stattype
    :bernoulli (bernoulli/export category)
    :categorical (categorical/export category)
    :gaussian (gaussian/export category)
    (throw (ex-info (str "pGPM doesn't exist: " stattype)
                    {:stattype stattype}))))

(defn ->pGPM
  "Cast a spec to the specified pGPM.
  Optionally takes suff-stats, hyperparameters, and options, by key."
  [primitive var-name & {:keys [suff-stats hyperparameters options]}]
  (case primitive
    :bernoulli (bernoulli/spec->bernoulli var-name :suff-stats suff-stats :hyperparameters hyperparameters)
    :categorical (categorical/spec->categorical var-name :suff-stats suff-stats :hyperparameters hyperparameters :options options)
    :gaussian (gaussian/spec->gaussian var-name :suff-stats suff-stats :hyperparameters hyperparameters)
    :poisson (poisson/spec->poisson var-name :suff-stats suff-stats :hyperparameters hyperparameters)
    (throw (ex-info (str "pGPM doesn't exist for var-name: " primitive " for " var-name)
                    {:primitive primitive}))))
