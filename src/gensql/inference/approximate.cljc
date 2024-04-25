(ns gensql.inference.approximate
  (:require [gensql.inference.gpm.proto :as gpm.proto]
            [gensql.inference.utils :as utils]))

(defn mutual-info
  "Given a GPM, estimates the mutual-information of `target-a` and `target-b`
  given `constraints` with `n-samples`."
  [this target-a target-b constraints n-samples]
  (let [joint-target (into target-a target-b)
        samples (repeatedly n-samples #(gpm.proto/simulate
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
                                                        (gpm.proto/logpdf
                                                         this
                                                         (select-keys sample target)
                                                         (nth constraint i)))
                                                      samples)))
        ;; TODO: will we get perf improvements if the run one map for all of the below?
        logpdf-a  (logpdf-estimate target-a)
        logpdf-b  (logpdf-estimate target-b)
        logpdf-ab (logpdf-estimate joint-target)]
    (- logpdf-ab (+ logpdf-a logpdf-b))))
