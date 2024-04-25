(ns gensql.inference.dpm
  "Functions operating on discriminative population models. All probabilistic
  models for which you can simulate and compute the log-PDF only for a set of
  dependent variables (most typically 1 column variable), and for which
  conditions have to be completely defined (i.e. the set of keys in the
  conditions is equal to the set of independent/input variables). These
  probabilistic models need to know which variables are input vs output."
  (:require [clojure.set :as set]))

(defprotocol DependentVariables
  (dependent-variables [dpm]))

(defprotocol IndependentVariables
  (independent-variables [dpm]))

(defn dpm?
  "Returns `true` if `x` is a discriminative population model."
  [x]
  (and (satisfies? DependentVariables x)
       (satisfies? IndependentVariables x)))

(defn ^:private assert-targets-dependent
  "Throws an exception any of the targets are not dependent variables."
  [dpm target-variables]
  (when-let [vars (seq (set/difference (set target-variables)
                                       (set (dependent-variables dpm))))]
    (throw (ex-info "Some target variables are not dependent."
                    {:cognitect.anomalies/category :cognitect.anomalies/incorrect
                     :non-dependent-vars vars
                     :dpm dpm}))))

(defn ^:private assert-ivars-conditioned
  "Throws an exception if any of the independent variables are not conditioned."
  [dpm constrained-variables]
  (when-let [unconditioned-ivars (seq (set/difference (set (independent-variables dpm))
                                                      (set constrained-variables)))]
    (throw (ex-info "Not all independent variables are conditioned."
                    {:cognitect.anomalies/category :cognitect.anomalies/incorrect
                     :unconditioned-independent-variables unconditioned-ivars
                     :dpm dpm}))))

(defn assert-args-valid
  "Asserts that `gensql.inference.gpm/logpdf` and
  `gensql.inference.gpm/constraints` can be called with the provided
  targeted and constrained variables."
  [dpm target-variables constrained-variables]
  (assert-targets-dependent dpm target-variables)
  (assert-ivars-conditioned dpm constrained-variables))
