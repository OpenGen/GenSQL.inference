(ns inferenceql.inference.gpm.conditioned-test
  (:require [clojure.test :refer [are deftest]]
            [inferenceql.inference.gpm :as gpm]
            [inferenceql.inference.gpm.proto :as gpm.proto]
            [inferenceql.inference.gpm.conditioned :as conditioned]))

(deftest variables
  (are [vars conditions] (let [model (reify
                                       gpm.proto/Variables
                                       (variables [_]
                                         vars))
                               conditioned-model (conditioned/condition model conditions)]
                           (= vars (gpm/variables model))
                           (= vars (gpm/variables conditioned-model)))
    [] {}
    [:x] {}
    [] {:x 0}
    [:x] {:x 0}
    [:x :y] {}
    [] {:x 0}
    [:x :y] {:x 0}
    [:x :y] {:y 1}
    [:x :y] {:x 0 :y 1}))

(deftest logpdf-targets
  (are [targets conditions]
      (let [model (reify gpm.proto/GPM
                    (logpdf [_ actual _]
                      actual))
            conditioned-model (conditioned/condition model conditions)]
        (= targets (gpm/logpdf conditioned-model targets conditions)))
    [] {}
    [:x] {}
    [] {:x 0}
    [:x] {:x 0}
    [:x :y] {}
    [] {:x 0}
    [:x :y] {:x 0}
    [:x :y] {:y 1}
    [:x :y] {:x 0 :y 1}))

(deftest logpdf-conditions
  (are [condition-conditions conditions expected]
      (let [model (reify gpm.proto/GPM
                    (logpdf [_ _ actual]
                      actual))
            conditioned-model (conditioned/condition model condition-conditions)]
        (= expected (gpm/logpdf conditioned-model [:x] conditions)))
    {} {} {}
    {} {:x 0} {:x 0}
    {:x 0} {} {:x 0}
    {:x 0} {:x 1} {:x 1}
    {:x 0} {:y 1} {:x 0 :y 1}
    {:y 1} {:x 0} {:x 0 :y 1}
    {:x 0 :z 2} {:y 1} {:x 0 :y 1 :z 2}
    {:x 0} {:y 1 :z 2} {:x 0 :y 1 :z 2}))

(deftest simulate-conditions
  (are [c1 c2 expected]
      (let [model (reify gpm.proto/GPM
                    (simulate [_ _ actual]
                      actual))
            conditioned-model (conditioned/condition model c1)]
        (= expected (gpm/simulate conditioned-model [:x] c2)))
    {} {} {}
    {} {:x 0} {:x 0}
    {:x 0} {} {:x 0}
    {:x 0} {:x 1} {:x 1}
    {:x 0} {:y 1} {:x 0 :y 1}
    {:y 1} {:x 0} {:x 0 :y 1}
    {:x 0 :z 2} {:y 1} {:x 0 :y 1 :z 2}
    {:x 0} {:y 1 :z 2} {:x 0 :y 1 :z 2}))

(deftest logpdf-condition
  (are [c1 c2 expected]
      (let [model (-> (reify gpm.proto/GPM
                        (simulate [_ _ actual]
                          actual))
                      (conditioned/condition c1))]
        (= expected (gpm/simulate model [:x] c2)))
    {} {} {}
    {} {:x 0} {:x 0}
    {:x 0} {} {:x 0}
    {:x 0} {:x 1} {:x 1}
    {:x 0} {:y 1} {:x 0 :y 1}
    {:y 1} {:x 0} {:x 0 :y 1}
    {:x 0 :z 2} {:y 1} {:x 0 :y 1 :z 2}
    {:x 0} {:y 1 :z 2} {:x 0 :y 1 :z 2}))

(deftest logpdf-condition-twice
  (are [c1 c2 c3 expected]
      (let [model (-> (reify gpm.proto/GPM
                        (simulate [_ _ actual]
                          actual))
                      (conditioned/condition c1)
                      (conditioned/condition c2))]
        (= expected (gpm/simulate model [:x] c3)))
    {} {} {} {}
    {:x 0} {} {} {:x 0}
    {} {:x 0} {} {:x 0}
    {} {} {:x 0} {:x 0}
    {:x 0} {:x 1} {} {:x 1}
    {:x 0} {} {:x 1} {:x 1}
    {} {:x 0} {:x 1} {:x 1}))

(deftest merged-conditions
  (are [c1 c2 expected]
    (let [model (conditioned/condition nil c1)
          conditioned-model (gpm.proto/condition model c2)]
      (= expected (:conditions conditioned-model)))
    {} {} {}
    {} {:x 0} {:x 0}
    {:x 0} {} {:x 0}
    {:x 0} {:x 1} {:x 1}
    {:x 0} {:y 1} {:x 0 :y 1}
    {:y 1} {:x 0} {:x 0 :y 1}
    {:x 0 :z 2} {:y 1} {:x 0 :y 1 :z 2}
    {:x 0} {:y 1 :z 2} {:x 0 :y 1 :z 2}))
