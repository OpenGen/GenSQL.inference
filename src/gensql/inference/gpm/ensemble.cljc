(ns gensql.inference.gpm.ensemble
  (:require [clojure.math :as math]
            [gensql.inference.gpm.conditioned :as conditioned]
            [gensql.inference.gpm.constrained :as constrained]
            [gensql.inference.gpm.proto :as gpm.proto]
            [gensql.inference.utils :as utils])
  #?(:clj (:import [java.util ArrayList]
           [org.apache.commons.math3.distribution EnumeratedDistribution]
           [org.apache.commons.math3.util Pair])))

(defn map->enumerated-distribution
  [m]
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
    (gpm.proto/variables (first gpms)))

  gpm.proto/Condition
  (condition [this conditions]
    (conditioned/condition this conditions))

  gpm.proto/Constrain
  (constrain [this event opts]
    (constrained/constrain this event opts)))

(defn ensemble
  [models]
  (when-not (pos? (count models))
    (throw (ex-info "Must provide at least one model" {})))
  (->Ensemble (vec models)))
