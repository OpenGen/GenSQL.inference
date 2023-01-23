(ns inferenceql.inference.gpm.ensemble
  (:require [clojure.math :as math]
            [inferenceql.inference.gpm.proto :as gpm.proto]
            [inferenceql.inference.utils :as utils])
  #?(:clj (:import [java.util ArrayList]
           [org.apache.commons.math3.distribution EnumeratedDistribution]
           [org.apache.commons.math3.util Pair])))

(defn map->enumerated-distribution
  [m]
  (when-not (every? (complement pos?) (vals m))
    (throw (ex-info "Weights must be negative" {:weights (vals m)})))
  (let [pairs (ArrayList.)]
    (doseq [[k v] m]
      (.add pairs (Pair. k (math/exp (double v)))))
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

  (logpdf [_ targets constraints]
    (let [logpdfs (map #(gpm.proto/logpdf % targets constraints) gpms)]
      (if (seq constraints)
        (utils/logmeanexp-weighted (map #(gpm.proto/logpdf % constraints {}) gpms) 
                                        logpdfs)
      (utils/logmeanexp logpdfs))))

  gpm.proto/Variables
  (variables [_]
    (gpm.proto/variables (first gpms))))

(defn ensemble
  [models]
  (when-not (pos? (count models))
    (throw (ex-info "Must provide at least one model" {})))
  (->Ensemble (vec models)))
