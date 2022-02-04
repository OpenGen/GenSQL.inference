(ns inferenceql.inference.gpm.ensemble
  (:import [java.util ArrayList]
           [org.apache.commons.math3.distribution EnumeratedDistribution]
           [org.apache.commons.math3.util Pair])
  (:require [inferenceql.inference.gpm.proto :as gpm.proto]))

(defn map->enumerated-distribution
  [m]
  (when-not (every? (complement pos?) (vals m))
    (throw (ex-info "Weights must be negative" {:weights (vals m)})))
  (let [pairs (ArrayList.)]
    (doseq [[k v] m]
      (.add pairs (Pair. k (Math/exp (double v)))))
    (EnumeratedDistribution. pairs)))

(defn weighted-sample
  [m]
  (let [ed (map->enumerated-distribution m)]
    (.sample ed)))

(defrecord Ensemble [gpms]
  gpm.proto/GPM
  (simulate [_ targets constraints]
    (let [gpm (if-not (seq constraints)
                (rand-nth gpms)
                (weighted-sample
                 (zipmap gpms
                         (map #(gpm.proto/logpdf % constraints {})
                              gpms))))]
      (gpm.proto/simulate gpm targets constraints)))

  gpm.proto/Variables
  (variables [_]
    (gpm.proto/variables (first gpms))))

(defn ensemble
  [models]
  (when-not (pos? (count models))
    (throw (ex-info "Must provide at least one model" {})))
  (->Ensemble (vec models)))
