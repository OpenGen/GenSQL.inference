(ns gensql.inference.gpm.multimixture.search.enumerative-search-test
  "Enumerative tests."
  (:require [clojure.test :as test :refer [deftest is]]
            [gensql.inference.gpm.multimixture.search.enumerative :as enumerative]))

;; Checks that all configurations are returned for a specified number of rows
;; and clusters.
(deftest rows->all-cluster-configurations
  (let [num-rows     2
        num-clusters 2
        single-row   1]
    (is (= `((0 0) (0 1) (1 0) (1 1))
           (enumerative/rows->all-cluster-configurations
            num-rows
            num-clusters)))
    ;; Number of rows is one.
    (is (= `((0) (1))
           (enumerative/rows->all-cluster-configurations
            single-row
            num-clusters)))))

;; Checks that the returned posterior probability is correct regardless of
;; cluster assignment.
(deftest cluster-config->beta-update
  ;; Walkthrough of the first test.
  ;; alpha'_0 = alpha_0 + #_true = 0.5 + 1 = 1.5
  ;; beta'_0  = beta_0  + #_false = 0.5 + 0 = 0.5
  ;; update_0 = P[diff-config] * alpha'_0 / (alpha'_0 + beta'_0)
  ;;          = 0.5 * 1.5 / 2
  ;;          = 0.375
  ;; alpha'_1 = alpha_1 + #_true = 0.5 + 0 = 0.5
  ;; beta'_1  = beta_1  + #_false = 0.5 + 1 = 1.5
  ;; update_1 = P[diff-config] * alpha'_1 / (alpha'_1 + beta'_1)
  ;;          = 0.5 * 0.5 / 2
  ;;          = 0.125
  (let [known-labels [true false]
        diff-config  [0 1]
        same-config  [0 0]
        num-clusters 2
        prob-config  0.5
        beta-params  {:alpha 0.5 :beta 0.5}]
    ;; Assignments contains both clusters.
    (is (= [0.375 0.125]
           (enumerative/cluster-config->beta-update
            known-labels
            diff-config
            num-clusters
            prob-config
            beta-params)))
    ;; Assignments contain one cluster.
    (is (= [0.25 0.25]
           (enumerative/cluster-config->beta-update
            known-labels
            same-config
            num-clusters
            prob-config
            beta-params)))))

;; Ensures correct access to probability table and calculation of
;; configuration probability, which weights the posterior probability for a
;; particular cluster.
(deftest cluster-configs->beta-updates
  ;; Involved walkthrough, review the walkthrough in
  ;; `cluster-config->beta-update` before continuing.
  ;;
  ;; P[config_0] = 0.5 * 0.75 = 0.375
  ;; alpha'_0 = 1.5, beta'_0 = 0.5
  ;; update_0 =  0.375 * 0.75 = 0.28125
  ;; alpha'_1 = 0.5, beta'_1 = 1.5
  ;; update_1 =  0.375 * 0.25 = 0.09375
  ;;
  ;; P[config_1] = 0.5 * 0.25 = 0.125
  ;; alpha'_0 = 1.5, beta'_0 = 1.5
  ;; update_0 =  0.125 * 0.5 = 0.0625
  ;; alpha'_1 = 0.5, beta'_1 = 0.5
  ;; update_1 =  0.125 * 0.5 = 0.0625
  (let [known-labels [true false]
        configs      [[0 1] [0 0]]
        probs        [[0.5 0.5] [0.25 0.75]]
        beta-params {:alpha 0.5 :beta 0.5}]
    (is (= `([0.28125 0.09375] [0.0625 0.0625])
           (enumerative/cluster-configs->beta-updates
            probs
            known-labels
            beta-params
            configs)))))

;; Smoke test for enumerative search.
(deftest enumerative-search
  ;; `search` sums across all configurations to get predictive probabilities,
  ;; then sums over c_k for each row, P[c_k | row] * P[True | c_k].
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
        actual       (enumerative/search
                      spec
                      "y"
                      known-rows
                      unknown-rows
                      beta-params)]
    ;; Checks that sum of residuals is less than the threshhold.
    (is (< (apply  + (map (comp #(max % (- %)) -) expected actual)) 0.01))))
