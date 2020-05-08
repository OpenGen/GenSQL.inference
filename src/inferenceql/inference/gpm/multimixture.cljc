(ns inferenceql.inference.gpm.multimixture
  (:require [metaprob.distributions :as dist]
            [metaprob.prelude :as mp]
            [metaprob.generative-functions :as g :refer [gen]]
            [inferenceql.inference.gpm.multimixture.multimixture :as mmix]
            [inferenceql.inference.utils :as utils]
            [inferenceql.inference.gpm.proto :as gpm-proto]))

(defn optimized-row-generator
  [spec]
  (let [row-generator (mmix/row-generator spec)]
    (g/make-generative-function
     row-generator
     (gen [partial-trace]
       (let [all-latents    (mmix/all-latents spec)
             all-traces     (mapv #(merge partial-trace %)
                                  all-latents)
             all-logscores  (mapv #(last (mp/infer-and-score :procedure row-generator
                                                             :observation-trace %))
                                  all-traces)
             all-scores (map mp/exp all-logscores)
             all-zeroes (every? #(== 0 %) all-scores)
             log-normalizer (if all-zeroes ##-Inf (dist/logsumexp all-logscores))
             score          log-normalizer
             categorical-params (if all-zeroes
                                  (mmix/uniform-categorical-params (count all-scores))
                                  (dist/normalize-numbers all-scores))]
         (gen []
           (let [i     (dist/categorical categorical-params)
                 trace (nth all-traces i)
                 v     (first (mp/infer-and-score :procedure row-generator
                                                  :observation-trace trace))]
             [v trace score])))))))

;; XXX Currently, assumes that the row generator of the mmix map is passed in.
(defrecord Multimixture
    [model]
  gpm-proto/GPM

  (logpdf [this targets constraints inputs]
    (let [constraint-addrs-vals        (mmix/with-row-values {} constraints)
          target-constraint-addrs-vals (mmix/with-row-values {}
                                         (merge targets
                                                constraints))
          row-generator                (optimized-row-generator model)

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

  (simulate [this targets constraints n-samples inputs]
    (let [constraint-addrs-vals (mmix/with-row-values {} constraints)
          generative-model      (optimized-row-generator model)
          gen-fn                #(let [[sample _ _] (mp/infer-and-score :procedure generative-model
                                                                        :observation-trace constraint-addrs-vals)]
                                   (select-keys sample targets))]
      (take n-samples (repeatedly gen-fn))))

  (mutual-information [this target-a target-b constraints n-samples]
    (let [joint-target (into target-a target-b)
          samples (gpm-proto/simulate
                   this
                   (cond-> joint-target
                     (vector? constraints)
                     (into constraints))
                   constraints
                   n-samples
                   {})
          constraint (if (map? constraints)
                       (repeat n-samples constraints)
                       (map #(select-keys % constraints)
                            samples))
          logpdf-estimate (fn [target]
                            (utils/average (map-indexed (fn [i sample]
                                                          (gpm-proto/logpdf
                                                           this
                                                           (select-keys sample target)
                                                           (nth constraint i)
                                                           {}))
                                                        samples)))
          ;; TODO: will we get perf improvements if the run one map for all of the below?
          logpdf-a  (logpdf-estimate target-a)
          logpdf-b  (logpdf-estimate target-b)
          logpdf-ab (logpdf-estimate joint-target)]
      (- logpdf-ab (+ logpdf-a logpdf-b)))))
