(ns inferenceql.inference.gpm.multimixture
  (:require [metaprob.prelude :as mp]
            [inferenceql.inference.gpm.multimixture.utils :as mmix.utils]
            [inferenceql.inference.utils :as utils]
            [inferenceql.inference.gpm.proto :as gpm-proto]))

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

  (mutual-information [this target-a target-b constraints n-samples]
    (let [joint-target (into target-a target-b)
          samples (repeatedly n-samples #(gpm-proto/simulate
                                          this
                                          (cond-> joint-target
                                            (vector? constraints)
                                            (into constraints))
                                          constraints))
          constraint (if (map? constraints)
                       (repeat n-samples constraints)
                       (map #(select-keys % constraints)
                            samples))
          logpdf-estimate (fn [target]
                            (utils/average (map-indexed (fn [i sample]
                                                          (gpm-proto/logpdf
                                                           this
                                                           (select-keys sample target)
                                                           (nth constraint i)))
                                                        samples)))
          ;; TODO: will we get perf improvements if the run one map for all of the below?
          logpdf-a  (logpdf-estimate target-a)
          logpdf-b  (logpdf-estimate target-b)
          logpdf-ab (logpdf-estimate joint-target)]
      (- logpdf-ab (+ logpdf-a logpdf-b)))))
