(ns gensql.inference.gpm.multimixture
  (:require [metaprob.prelude :as mp]
            [gensql.inference.gpm.conditioned :as conditioned]
            [gensql.inference.gpm.constrained :as constrained]
            [gensql.inference.gpm.multimixture.utils :as mmix.utils]
            [gensql.inference.gpm.proto :as gpm-proto]))

;; XXX Currently, assumes that the row generator of the mmix map is passed in.
(defrecord Multimixture [vars views]
  gpm-proto/GPM

  (logpdf [this targets constraints]
    (let [constraint-addrs-vals        (mmix.utils/with-row-values {} constraints)
          target-constraint-addrs-vals (mmix.utils/with-row-values {}
                                         (merge targets
                                                constraints))
          row-generator                (mmix.utils/optimized-row-generator this)

          ;; Run infer to obtain probabilities.
          [_ _ log-weight-numer] (mp/infer-and-score
                                  :procedure row-generator
                                  :observation-trace target-constraint-addrs-vals)
          log-weight-denom (if (empty? constraint-addrs-vals)
                             ;; There are no constraints: log weight is zero.
                             0
                             ;; There are constraints: find marginal probability of constraints.
                             (let [[_ _ weight] (mp/infer-and-score
                                                 :procedure row-generator
                                                 :observation-trace constraint-addrs-vals)]
                               weight))]
      (- log-weight-numer log-weight-denom)))

  (simulate [this targets constraints]
    (let [constraint-addrs-vals (mmix.utils/with-row-values {} constraints)
          generative-model      (mmix.utils/optimized-row-generator this)
          [sample _ _] (mp/infer-and-score :procedure generative-model :observation-trace constraint-addrs-vals)]
      (select-keys sample targets)))

  gpm-proto/Variables
  (variables [this]
    (set (keys (get this :vars))))

  gpm-proto/Condition
  (condition [this conditions]
    (conditioned/condition this conditions))

  gpm-proto/Constrain
  (constrain [this event opts]
    (constrained/constrain this event opts)))
