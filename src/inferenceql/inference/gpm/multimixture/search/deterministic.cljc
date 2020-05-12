(ns inferenceql.inference.gpm.multimixture.search.deterministic
  (:require [inferenceql.inference.utils :as utils]
            [inferenceql.inference.gpm.multimixture.search.utils :as search.utils]))

#?(:cljs (enable-console-print!))

(defn update-beta-params
  "Updates beta params for each component, within each view."
  [beta-params known-rows new-column-key num-clusters known-probs]
  (let [obs-probs-pairs (map vector known-probs (map #(get % new-column-key) known-rows))
        true-obs        (filter #(second %) obs-probs-pairs)
        false-obs       (filter #(not (second %)) obs-probs-pairs)
        obs-func        (fn [obs param]
                           ;; If no observations, alpha and beta do not get updated.
                          (if (empty? obs)
                            (repeat num-clusters {param (param beta-params)})
                            (->> obs
                                  ;; Ignore indices and group by cluster, not row, to compute
                                  ;; the necessary sum.
                                 (map #(first %))
                                 (utils/transpose)
                                 (map #(apply + %))
                                 (map (fn [param-instance]
                                        {param (+ (param beta-params) param-instance)})))))
        alphas (obs-func true-obs :alpha)
        betas (obs-func false-obs :beta)]
    (map merge alphas betas)))

(defn search
  "Mimicks the behavior of search, but without sampling!"
  [spec new-column-key known-rows unknown-rows beta-params]
  (let [[known-probs unknown-probs] (search.utils/generate-cluster-row-probability-table spec known-rows unknown-rows)
        num-clusters               (count (get-in spec [:views 0]))
        beta-primes                (map #(update-beta-params
                                          beta-params
                                          known-rows
                                          new-column-key
                                          num-clusters
                                          %)
                                        known-probs)]
    (vec (flatten (map-indexed
                   (fn [row-idx _]
                     (map-indexed
                      (fn [view-idx beta-prime]
                        (reduce +
                                (map-indexed
                                 (fn [idx-cluster cluster-params]
                                   (let [row-cluster-prob (-> unknown-probs
                                                              (nth view-idx)
                                                              (nth row-idx)
                                                              (nth idx-cluster))]
                                     (* row-cluster-prob
                                        (/ (:alpha cluster-params)
                                           (+ (:alpha cluster-params)
                                              (:beta cluster-params))))))
                                 beta-prime)))
                      beta-primes))
                   unknown-rows)))))
