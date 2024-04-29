(ns gensql.inference.gpm.multimixture.search.utils
  (:require [gensql.inference.gpm.multimixture.utils :as mmix.utils]
            [gensql.inference.utils :as utils]
            [metaprob.prelude :as mp]))

(defn normalize-row-probability
  "Normalizes a collection of non-negative numbers."
  [coll]
  (let [z (reduce + coll)]
    (mapv #(/ % z) coll)))

(defn cluster-row-probability
  "Determines the probability that a specific cluster component generated the given row."
  [spec cluster-idx view-idx row]
   ;; Prior probability * likelihood that cluster generated row, given the row and spec.
  (let [cluster (get-in spec [:views view-idx cluster-idx])]
    (* (:probability cluster)
       (mp/exp (last (mp/infer-and-score :procedure (mmix.utils/cluster-row-generator cluster (get spec :vars))
                                         :observation-trace (mmix.utils/with-row-values {} row)))))))

(defn view-row-probabilities
  "Returns a probability table P, where
   P[row][cluster-component] = normalized probability (within the specified view)
                               that `cluster-component` generated `row`."
  [spec rows]
  (map-indexed (fn [view-idx view]
                 ;; For each cluster in a view, for each row in cluster, determine the
                 ;; probability that a cluster generated a row.
                 (->> view
                      (map-indexed (fn [cluster-idx _]
                                     (map #(cluster-row-probability
                                            spec
                                            cluster-idx
                                            view-idx
                                            %) rows)))
                      ;; Group probabilities by row, rather than by cluster.
                      (utils/transpose)
                      (map normalize-row-probability)))
               (get spec :views)))

(defn generate-cluster-row-probability-table
  "Returns table P where, P[known/unknown][view][row][component] is equivalent
   to the probability that a known/unknown row within a view by a specific component."
  [spec known-rows unknown-rows]
  (map (fn [rows]
         (view-row-probabilities spec rows)) [known-rows unknown-rows]))
