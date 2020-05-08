(ns inferenceql.inference.gpm.multimixture.search.enumerative
  (:require [clojure.math.combinatorics :as combo]
            [inferenceql.inference.gpm.multimixture.utils :as mmix-utils]
            [inferenceql.inference.gpm.multimixture.search.utils :as utils]))

(defn rows->all-cluster-configurations
  "Returns a (lazy) sequence of all possible cluster configurations for rows."
  [num-rows num-clusters]
  (if (> num-rows 1)
    (map flatten
         (reduce combo/cartesian-product
                 (repeatedly num-rows
                             #(range num-clusters))))
    (combo/cartesian-product (range num-clusters))))

(defn cluster-config->beta-update
  "Given a cluster configuration and its probability, calculates the
  weighted beta parameter update."
  [known-labels config num-clusters prob-config beta-params]
  (let [freq (frequencies (map vector config known-labels))
        alpha (:alpha beta-params)
        beta  (:beta  beta-params)]
    (mapv (fn [cluster]
            (let [[alpha' beta'] (map + [alpha beta]
                                      [(get freq `[~cluster  true] 0)
                                       (get freq `[~cluster false] 0)])]
              (* prob-config (/ alpha' (+ alpha' beta')))))
          (range num-clusters))))

(defn cluster-configs->beta-updates
  "Given a probability table in the form P[row][cluster] and a list of cluster
  configurations, returns the list of respective probabilities of said configurations
  multiplied by the beta-parameter update."
  [probs known-labels beta-params configs]
  (let [num-clusters (count (first probs))]
    (map (fn [config]
           ;; Calculate probability of each assignment and take the product.
           (let [prob-config (reduce * (map-indexed
                                        (fn [row assignment]
                                          (get-in probs [row assignment]))
                                        config))]
             ;; Get individual beta parameter update for configuration.
             (cluster-config->beta-update
              known-labels
              config
              num-clusters
              prob-config
              beta-params)))
         configs)))

(defn search
  "Mimicks the behavior of search, but without sampling, and it's enumerative!
  Assumes one view in the specification."
  [spec new-column-key known-rows unknown-rows beta-params]
  (let [[known-probs unknown-probs] (utils/generate-cluster-row-probability-table spec known-rows unknown-rows)
        num-clusters                (count (get-in spec [:views 0]))
        known-probs-single-view     (vec (nth   known-probs 0)) ;; hard coded for one view.
        unknown-probs-single-view   (vec (nth unknown-probs 0)) ;; hard coded for one view.
        known-labels                (mapv #(get % new-column-key) known-rows)
        thetas                      (->> (rows->all-cluster-configurations (count known-rows) num-clusters)
                                         (cluster-configs->beta-updates
                                          known-probs-single-view
                                          known-labels
                                          beta-params)
                                         (mmix-utils/transpose)
                                         (map #(apply + %)))]
    (map-indexed (fn [row _]
                   ;; Get prediction by multiplying likelihood of cluster by coin weight.
                   (apply + (map * (get unknown-probs-single-view row) thetas)))
                 unknown-rows)))
