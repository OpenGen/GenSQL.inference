(ns inferenceql.inference.gpm.multimixture.search.gibbs
  (:require [metaprob.distributions :as dist]
            [inferenceql.inference.gpm.multimixture.search.utils :as utils]))

(defn sample-cluster-assignments
  "Given a probability table for rows, samples cluster assignments. A valid
  table is of the form P[view][row][cluster]. Currently assumes one view, so
  the table should be of the form P[row][cluster]."
  [probs]
  (let [n-rows (count probs)]
    (mapv #(-> probs
               (nth %)
               (dist/categorical))
          (range n-rows))))

(defn cluster-assignments->thetas
  "For each cluster c_1, ..., c_k, computes the posterior probability on that cluster,
  given cluster assignments, and known labels."
  [assignments beta-params n-clusters labels]
  ;; `count-map` is of the form {c_i {true n-true false n-false}} for all clusters c_i.
  (let [count-map (->> (mapv (fn [c r] [c r]) assignments labels)
                       (frequencies)
                       (reduce-kv (fn [m k v]
                                    (let [[cluster label] k]
                                      (update-in m [cluster label] (fnil #(+ % v) 0))))
                                  {}))]
    ;; theta = a' / (a' + b'), a' = # of true labels + a, b' = # of false labels + b
    (mapv (fn [c]
            (let [a' (+ (get-in count-map [c true]  0) (:alpha beta-params))
                  b' (+ (get-in count-map [c false] 0) (:beta  beta-params))]
              (/ a' (+ a' b'))))
          (range n-clusters))))

(defn thetas->pred-probs
  "Given cluster assignments of unknown clusters, as well as cluster posterior probabilities,
  returns the corresponding posterior probability."
  [unknown-clusters thetas]
  (map #(nth thetas %) unknown-clusters))

(defn search
  "Mimicks the behavior of search, but without sampling, and it's approximated with Gibbs sampling!
  Assumes one view in the specification."
  ([spec new-column-key known-rows unknown-rows beta-params]
    (search spec new-column-key known-rows unknown-rows beta-params 1000))
  ([spec new-column-key known-rows unknown-rows beta-params iters]
   (let [[known-probs unknown-probs] (utils/generate-cluster-row-probability-table
                                      spec
                                      known-rows
                                      unknown-rows)
         unknown-probs-single-view   (nth unknown-probs 0)
         known-probs-single-view     (nth   known-probs 0)
         n-clusters                  (count (get-in spec [:views 0]))
         known-labels                (mapv #(get % new-column-key) known-rows)
         pred                        (fn [known-clusters unknown-clusters]
                                       (->> known-labels
                                            (cluster-assignments->thetas
                                             known-clusters
                                             beta-params
                                             n-clusters)
                                            (thetas->pred-probs unknown-clusters)))]
     (loop [unknown-clusters (sample-cluster-assignments unknown-probs-single-view)
            known-clusters   (sample-cluster-assignments known-probs-single-view)
            pred-probs       (pred known-clusters unknown-clusters)
            n-iters          iters]
       (if (= n-iters 0)
         (map #(/ % iters) pred-probs)  ;; Return the final result, empirical average.
         (let [new-unknown-clusters (sample-cluster-assignments unknown-probs-single-view)
               new-known-clusters   (sample-cluster-assignments   known-probs-single-view)]
           (recur new-unknown-clusters
                  new-known-clusters
                  (map + pred-probs (pred new-known-clusters new-unknown-clusters))
                  (dec n-iters))))))))
