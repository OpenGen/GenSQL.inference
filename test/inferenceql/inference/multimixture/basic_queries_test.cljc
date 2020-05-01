(ns inferenceql.inference.multimixture.basic-queries-test
  (:require [clojure.spec.alpha :as s]
            [clojure.test :as test :refer [deftest testing is]]
            [clojure.walk :as walk :refer [stringify-keys]]
            #?(:clj [clojure.string :as str])
            [expound.alpha :as expound]
            #?(:clj [inferenceql.inference.plotting.generate-vljson :as plot])
            [inferenceql.inference.utils :as utils]
            [inferenceql.inference.multimixture.specification :as spec]
            [inferenceql.inference.multimixture.search :as search] ;; XXX: why is the "optimized" row generator in search?
            [inferenceql.inference.gpm :as gpm]
            [metaprob.distributions :as dist]))

;; The following data generator has some interesting properties:
;; - clusters 0 and 1 in view 0 share the samme mu parameter.
;; - a is a deterministic indicator of the cluster.
;; - b is a noisy copy of a.
;; - in both views, clusters are equally weighted.
;; - in view 1, the third Gaussian components (cluster 0) "spans" the domain of
;; all the other components and share a center with cluster 1.
;;
;; I'd encourage everyone who works with the file to run the tests in this file
;; and then run `make plots` to see how the components relate.
(def multi-mixture
  {:vars {"x" :gaussian
          "y" :gaussian
          "z" :gaussian
          "a" :categorical
          "b" :categorical
          "c" :categorical}
   :views [[  {:probability 0.1666666666666
               :parameters {"x" {:mu 3 :sigma 1}
                            "y" {:mu 4 :sigma 0.1}
                            "a" {"0" 1.0 "1" 0.0 "2" 0.0 "3" 0.0 "4" 0.0 "5" 0.0}
                            "b" {"0" 0.95, "1" 0.01, "2" 0.01, "3" 0.01, "4" 0.01, "5" 0.01}}}
            {:probability  0.1666666666666
             :parameters {"x" {:mu 3 :sigma 0.1}
                          "y" {:mu 4 :sigma 1}
                          "a" {"0" 0.0 "1" 1.0 "2" 0.0 "3" 0.0 "4" 0.0 "5" 0.0}
                          "b" {"0" 0.01, "1" 0.95, "2" 0.01, "3" 0.01, "4" 0.01, "5" 0.01}}}
            {:probability 0.1666666666666
             :parameters {"x" {:mu 8  :sigma 0.5}
                          "y" {:mu 10 :sigma 1}
                          "a" {"0" 0.0 "1" 0.0 "2" 1.0 "3" 0.0 "4" 0.0 "5" 0.0}
                          "b" {"0" 0.01, "1" 0.01, "2" 0.95, "3" 0.01, "4" 0.01, "5" 0.01}}}
            {:probability 0.1666666666666
             :parameters {"x" {:mu 14  :sigma 0.5}
                          "y" {:mu  7  :sigma 0.5}
                          "a" {"0" 0.0 "1" 0.0 "2" 0.0 "3" 1.0 "4" 0.0 "5" 0.0}
                          "b" {"0" 0.01, "1" 0.01, "2" 0.01, "3" 0.95, "4" 0.01, "5" 0.01}}}
            {:probability 0.1666666666666
             :parameters {"x" {:mu 16  :sigma 0.5}
                          "y" {:mu  9  :sigma 0.5}
                          "a" {"0" 0.0 "1" 0.0 "2" 0.0 "3" 0.0 "4" 1.0 "5" 0.0}
                          "b" {"0" 0.01, "1" 0.01, "2" 0.01, "3" 0.01, "4" 0.95, "5" 0.01}}}
            {:probability 0.16666666666667
             :parameters {"x" {:mu  9  :sigma 2.5}
                          "y" {:mu 16  :sigma 0.1}
                          "a" {"0" 0.0 "1" 0.0 "2" 0.0 "3" 0.0 "4" 0.0 "5" 1.0}
                          "b" {"0" 0.01, "1" 0.01, "2" 0.01, "3" 0.01, "4" 0.01, "5" 0.95}}}]
           [{:probability 0.25
             :parameters {"z" {:mu 0 :sigma 1}
                          "c" {"0" 1.0, "1" 0.0, "2" 0.0, "3" 0.0}}}
            {:probability 0.25
             :parameters {"z" {:mu 15 :sigma 1}
                          "c" {"0" 0.0, "1" 1.0, "2" 0.0, "3" 0.0}}}
            {:probability 0.25
             :parameters {"z" {:mu 30 :sigma 1}
                          "c" {"0" 0.0, "1" 0.0, "2" 1.0, "3" 0.0}}}
            {:probability 0.25
             :parameters {"z" {:mu 15 :sigma 8}
                          "c" {"0" 0.0, "1" 0.0, "2" 0.0, "3" 1.0}}}]]})

(deftest multi-mixture-is-valid
  (when-not (s/valid? ::spec/multi-mixture multi-mixture)
    (expound/expound ::spec/multi-mixture multi-mixture))
  (is (s/valid? ::spec/multi-mixture multi-mixture)))

;; XXX: the test points are still defined using keys instead of strings.
(def test-points
  [{:x 3  :y 4}
   {:x 8  :y 10}
   {:x 14 :y 7}
   {:x 15 :y 8}
   {:x 16 :y 9}
   {:x 9  :y 16}])

(def cluster-point-mapping
  ;; Maps each cluster index to the ID of the point that is at the cluster's
  ;; center. Note that not all points have a cluster around them.
  {0 1
   1 1
   2 2
   3 3
   ;; P4 between clusters 4 and 5
   4 5
   5 6})
(def point-cluster-mapping {1 #{0 1}, 2 #{2}, 3 #{3}, 4 #{3 4}, 5 #{4}, 6 #{5}})
(def points-unique-cluster-mapping (select-keys
                                    point-cluster-mapping
                                    (for [[k v] point-cluster-mapping
                                          :when (= (count v) 1)] k)))
(def points-two-cluster-mapping (select-keys
                                 point-cluster-mapping
                                 (for [[k v] point-cluster-mapping
                                       :when (= (count v) 2)] k)))

(defn test-point
  "Retrieves a given point given its ID. Note that point IDs are different from
  their indexes in `test-points`: Point IDs are 1-indexed."
  [point-id]
  (nth test-points (dec point-id)))

(defn invert-map
  "Reverse the keys/values of a map"
  [m]
  (reduce-kv (fn [m k v]
               (update m v (fnil conj #{}) k))
             {}
             m))

(defn euclidean-distance
  [p1 p2]
  (Math/sqrt (->> (map - p1 p2)
                  (map #(Math/pow % 2))
                  (reduce +))))

(deftest points-equidistant-from-cluster-centers
  (doseq [[point-id clusters] (invert-map cluster-point-mapping)]
    (testing (str "Each cluster in " clusters " should be an equal distance from P" point-id)
      ;; Subsequent tests rely on this property of the test multimixture.
      (let [{:keys [x y] :as _point} (test-point point-id)
            cluster-center (fn [cluster]
                             [(spec/mu multi-mixture "x" cluster)
                              (spec/mu multi-mixture "y" cluster)])
            distances (->> clusters
                           (map cluster-center)
                           (map #(euclidean-distance [x y] %)))]
        (is (apply = distances))))))

;; XXX When I ported this, I got really derailed, because the whole machinery
;; below relies on variables not being all variables in the spec, but just the
;; variables in the first view.
(def variables (spec/view-variables (first (:views multi-mixture))))

(def numerical-variables
  (into #{}
        (filter #(spec/numerical? multi-mixture %))
        variables))
(def categorical-variables
  (into #{}
        (filter #(spec/nominal? multi-mixture %))
        variables))

(deftest test-cluster-point-mapping
  ;; This test verifies that we've constructed our multi-mixture correctly such
  ;; that the subsequent tests can succeed. Clusters that have a point at their
  ;; center should have that point's variables as the mu for each gaussian that
  ;; makes up the cluster gaussian that makes up the cluster.
  (doseq [[cluster point-id] cluster-point-mapping]
    (doseq [variable #{:x :y}]
      (let [point-value (get (test-point point-id) variable)
            mu (spec/mu multi-mixture (name variable) cluster)]
        (is (= point-value mu))))))

;; Define the row-generator used below.
(def row-generator (search/optimized-row-generator multi-mixture))

;; Define the MMix GPM.
(def gpm-mmix (gpm/Multimixture multi-mixture))

;; Some smoke tests. Because the code for the tests didn't run anymore as state
;; lost in a squash-merge, I needed those smoke tests to gain confidence on all
;; datastructure beeing correct.
(deftest test-smoke-row-generator
  (is (map? (row-generator))))

(deftest test-smoke-simulate
  (is (= 3 (count (gpm/simulate gpm-mmix {} {} 3)))))

(deftest test-smoke-simulate-conditional
  (is (= 999. (get (first (gpm/simulate gpm-mmix {} {"x" 999.} 3))
                   "x"))))

(deftest test-smoke-logpdf
  (is (float? (gpm/logpdf gpm-mmix {"x" 0.} {"y" 1.}))))

(def plot-point-count 1000)
;; The purpose of this test is to help the reader understand the test suite. It
;; generates Vega-Lite JSON as a side effect which can be rendered into plots.
;; See https://github.com/probcomp/inferenceql/issues/81 for why it is
;; Clojure-only.
#?(:clj (deftest visual-test
          ;; This tests saves plots for all simulated data in out/json results/
          ;; Plots can be generated with `make plots`.
          (testing "(smoke) simulate n complete rows and save them as vl-json"
            (let [num-samples plot-point-count
                  point-data (map-indexed (fn [index point]
                                            (reduce-kv (fn [m k v]
                                                         (assoc m (keyword (str "t" (name k))) v))
                                                       {:test-point (str "P " (inc index))}
                                                       point))
                                          test-points)
                  samples (take plot-point-count (repeatedly row-generator))]
              (utils/save-json "simulations-x-y"
                               (plot/scatter-plot-json ["x" "y"]
                                                       samples
                                                       point-data
                                                       [0 18]
                                                       "View 1: X, Y, A, B"))
              (utils/save-json "simulations-z"
                               (plot/hist-plot (utils/column-subset samples ["z" "c"])
                                               ["z" "c"]
                                               "Dim Z and C"))
              (doseq [variable #{"a" "b" "c"}]
                (utils/save-json (str "simulations-" (name variable))
                                 (plot/bar-plot (utils/column-subset samples [variable])
                                                (str "Dim " (-> variable name str/upper-case))
                                                plot-point-count)))
              (is (= (count samples) plot-point-count))))))


(def simulation-count 100)
(def threshold 0.5)

(defn- almost-equal? [a b] (utils/almost-equal? a b utils/relerr threshold))
(defn- almost-equal-vectors? [a b] (utils/almost-equal-vectors? a b utils/relerr threshold))
(defn- almost-equal-maps? [a b] (utils/almost-equal-maps? a b utils/relerr threshold))
(defn- almost-equal-p? [a b] (utils/almost-equal? a b utils/relerr 0.01))

(deftest simulations-conditioned-on-determinstic-category
  (doseq [[cluster point-id] cluster-point-mapping]
    (testing (str "Conditioned on  deterministic category" cluster)
      ;; We simulate all variables together in a single test like this because
      ;; there's currently a performance benefit to doing so.
      (let [point   (test-point point-id)
            samples (gpm/simulate gpm-mmix {} {"a" (str cluster)} simulation-count)]
        (doseq [variable variables]
          (cond (spec/numerical? multi-mixture variable)
                (let [samples (utils/col variable samples)]
                  (testing (str "validate variable " variable)
                    (let [sigma (spec/sigma multi-mixture variable cluster)]
                      (testing "mean"
                        (is (utils/almost-equal? (get point (keyword variable))
                                                 (utils/average samples)
                                                 utils/relerr
                                                 (/ sigma 2))))
                      (testing "standard deviation"
                        (is (utils/within-factor?
                              sigma
                              (utils/std samples)
                              2)))))
                  (spec/nominal? multi-mixture variable))))))))

(defn- true-categorical-p
  [point-cluster-mapping point]
  (let [possible-clusters (get point-cluster-mapping point)]
    (into {} (map (fn [cluster] [(str cluster)
                                 (if (contains? possible-clusters cluster)
                                   (/ 1 (count possible-clusters))
                                   0.)])
                  (range 6)))))

;; Below, we're making use of the fact that each value of "a" determines a
;; cluster.
(deftest simulations-conditioned-on-points
  ;; Tests that if we simulate conditioned on the test points we are simulating
  ;; from the right clusters.
  (doseq [[point-id clusters] (invert-map cluster-point-mapping)]
    (testing (str "Conditioned on point P" point-id)
      (let [point   (stringify-keys (test-point point-id))
            samples (gpm/simulate gpm-mmix {} point simulation-count)]
        (testing "validate cluster assignments/categorical distribution"
          (let [samples-a          (utils/column-subset samples ["a"])
                cluster-p-fraction (utils/probability-for-categories samples-a (map str (range 6)))
                true-p-category    (true-categorical-p point-cluster-mapping point-id)]
            (is (almost-equal-maps? true-p-category cluster-p-fraction))))))))

(deftest logpdf-numerical-given-categorical
  ;; Test logpdf of test points given the categorical that maps to clusters.
  (doseq [[cluster point-id] cluster-point-mapping]
    (let [point (stringify-keys
                 (select-keys (test-point point-id) (map keyword numerical-variables)))
          analytical-logpdf (transduce (map (fn [variable]
                                              (let [mu (spec/mu multi-mixture variable cluster)
                                                    sigma (spec/sigma multi-mixture variable cluster)]
                                                (dist/score-gaussian (get point variable) [mu sigma]))))
                                       +
                                       numerical-variables)
          queried-logpdf (gpm/logpdf gpm-mmix point {"a" (str cluster)})]
      (is (almost-equal-p? analytical-logpdf queried-logpdf)))))

;; XXX -- not sure what to do with the next two tests. I have added to gain
;; evidence that logPDF works for categoricals conditioned on numerical variables.
;; There more complicated machinery for the larger mmix model did not do that
;; for me. I don't know where those tests should go.
;; They are not documented in the README

(deftest logpdf-categoricals-given-point-one-component-model
  ;; Tests a 3 dimensional, single component model.
  (let [mmix-simple {:vars {"x" :gaussian
                            "a" :categorical
                            "b" :categorical}
                     :views [[{:probability 1.
                               :parameters {"x" {:mu 3 :sigma 1}
                                            "a" {"0" 1.0 "1" 0.0}
                                            "b" {"0" 0.95, "1" 0.05}}}]]}]
    (is (= 0.95  (Math/exp (gpm/logpdf
                            (gpm/Multimixture mmix-simple)
                            {"b" "0"}
                            {"x" 3.}))))))

;;; XXX -- same as above.
(deftest logpdf-categoricals-given-point-two-component-mix
  ;; Tests a 3 dimensional, two component model.
  (let [mmix-simple {:vars {"x" :gaussian
                            "a" :categorical
                            "b" :categorical}
                     :views [[{:probability 0.95
                               :parameters {"x" {:mu 3 :sigma 1}
                                            "a" {"0" 1.0 "1" 0.0}
                                            "b" {"0" 1.0, "1" 0.0}}}
                              {:probability 0.05
                               :parameters {"x" {:mu 3 :sigma 1}
                                            "a" {"0" 1.0 "1" 0.0}
                                            "b" {"0" 0.0 "1" 1.0 }}}]]}]
    (is (almost-equal-p? 0.95 (Math/exp (gpm/logpdf
                                         (gpm/Multimixture mmix-simple)
                                         {"b" "0"}
                                         {"x" 3.}
                                         ))))))

;; Define categories that are possible for "a" and "b". Relies on the assumption
;; that "a" and "b" have the same categories.
(def categories
  (keys
   (get (:parameters (first (first (multi-mixture :views)))) "a")))

(deftest logpdf-categoricals-given-points-that-identify-unique-cluster
  ;; Test logpdf of categical "b" given test points that uniquely map to one
  ;; cluster.
  (doseq [[point-id cluster-set] points-unique-cluster-mapping]
    (doseq [category categories]
      (testing (str "Point " point-id " Observing b=" category)
        (let [point          (stringify-keys (test-point point-id))
              cluster        (first cluster-set)
              analytical-pdf (get
                               (get (:parameters (nth (first (:views multi-mixture )) cluster)) "b") category)
              queried-pdf    (Math/exp (gpm/logpdf
                                        gpm-mmix
                                        {"b" category}
                                        point))]
          (is (almost-equal-p? analytical-pdf queried-pdf)))))))


(deftest logpdf-categoricals-given-points-that-belong-to-two-clusters
  ;; Test logpdf of categical "a" given test points that map to two clusters.
  (doseq [[point-id cluster-set] points-two-cluster-mapping]
    (doseq [category categories]
      (testing (str "Point " point-id " Observing a=" category)
        (let [point (stringify-keys (test-point point-id))
              analytical-pdf (if (contains? (set (map str cluster-set)) category)
                               0.5 ;; There are exactly two clusters that are equally
                               ;; to have generated this observation.
                               0   ;; No component is likely to have generated this observation
                               )
              queried-pdf   (Math/exp (gpm/logpdf
                                       gpm-mmix
                                       {"a" category}
                                       point))]
          (is (almost-equal-p? analytical-pdf queried-pdf)))))))
