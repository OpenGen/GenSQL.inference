(ns inferenceql.inference.gpm.multimixture.search.gibbs-search-test
  "Gibbs sampling tests."
  (:require [clojure.test :as test :refer [deftest is]]
            [inferenceql.inference.gpm.multimixture.utils :as mmix-utils]
            [inferenceql.inference.gpm.multimixture.search.gibbs :as gibbs]))

;; Checks that sampled values of clusters are within reason.
(deftest sample-cluster-assignments
  (let [probs      [[0.1 0.3 0.6]
                    [0.4 0.2 0.4]]
        n-clusters (count (first probs))
        total-el   (* n-clusters (count probs))
        sampled    (->> (fn [] (gibbs/sample-cluster-assignments probs))
                        (repeatedly 1000)
                        (mmix-utils/transpose)
                        (mapv #(->> (frequencies %)
                                    (into (sorted-map))
                                    (mapv second)))
                        (mapv #(let [sum (apply + %)]
                                 (mapv (fn [el] (/ el sum)) %))))
        ;; Sums the absolute residuals between empirical distribution
        ;; and true distribution, and checks that it is below the
        ;; threshhold.
        residuals (->> (mapcat #(map - %1 %2)
                               probs
                               sampled)
                       (map #(Math/abs %)))]
    (doseq [residual residuals]
      (is (< residual 0.05)))
    (is (< (/ (apply + residuals)
              total-el)
           0.05))))

;; Checks that given cluster assignments for all rows, the posterior
;; probabilities are correctly calculated.
(deftest cluster-assignments->thetas
  (let [diff-assign  [0 1]
        same-assign  [0 0]
        beta-params  {:alpha 0.5 :beta 0.5}
        n-clusters   2
        labels       [true false]
        diff-thetas  (gibbs/cluster-assignments->thetas
                      diff-assign
                      beta-params
                      n-clusters
                      labels)
        same-thetas  (gibbs/cluster-assignments->thetas
                      same-assign
                      beta-params
                      n-clusters
                      labels)]
    ;; Assignments contains both clusters.
    (is (= [0.75 0.25] diff-thetas))
    ;; Assignments contain one cluster.
    (is (= [0.50 0.50] same-thetas))))

;; Verifies the mapping from cluster configuration to cluster posterior.
(deftest thetas->pred-probs
  (let [unknown-clusters [0 1 0]
        thetas           [0.3 0.7]]
    (is (= [0.3 0.7 0.3] (gibbs/thetas->pred-probs
                          unknown-clusters
                          thetas)))))

;; Smoke test for Gibbs row sampling search.
(deftest gibbs-search
  ;; Gibbs search works as the following.
  ;;  0. Generate probability table given the spec and rows.
  ;;  1. For iter = 1 .. iters:
  ;;    a. Resample cluster assignments for all rows.
  ;;        c_i ~ Categorical(prob-table[row_i])
  ;;    b. Calculate predictive probabilities based on updates.
  ;;        alpha_k' = alpha_k + # True obs. in cluster k
  ;;        beta_k'  = beta_k  + # False obs. in cluster k
  ;;        pred_prob = alpha_k' / (alpha_k' + beta_k')
  ;;    c. Record pred. prob.
  ;;  2. Return sum(pred. probs) / iters.
  (let [spec          {:vars {"x" :gaussian}
                       :views [[{:probability 0.5
                                 :parameters {"x" {:mu 1 :sigma 1}}}
                                {:probability 0.5
                                 :parameters {"x" {:mu 4 :sigma 1}}}]]}
        unknown-rows [{"x" 0}
                      {"x" 5}]
        known-rows   [{"x" 0 "y" true}
                      {"x" 5 "y" false}]
        beta-params   {:alpha 0.5 :beta 0.5}
        expected     [0.75 0.25]
        actual       (gibbs/search
                      spec
                      "y"
                      known-rows
                      unknown-rows
                      beta-params)]
    ;; Checks that sum of residuals is less than the threshhold.
    (is (< (apply + (map (comp #(max % (- %)) -) expected actual)) 0.01))))
