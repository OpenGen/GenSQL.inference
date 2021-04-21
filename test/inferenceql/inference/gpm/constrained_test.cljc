(ns inferenceql.inference.gpm.constrained-test
  (:require [clojure.test :refer [are deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [inferenceql.inference.gpm :as gpm]
            [inferenceql.inference.gpm.constrained :as constrained]
            [inferenceql.inference.gpm.proto :as gpm.proto]
            [inferenceql.inference.gpm.primitive-gpms.categorical :as categorical]
            [metaprob.distributions :as distributions]
            [net.cgrand.xforms.rfs :as rfs]))

(def mmix
  (gpm/Multimixture
   {:vars {:x :categorical
           :y :gaussian}
    :views [[{:probability 0.75
              :parameters  {:x {"yes" 1.0 "no" 0.0}
                            :y {:mu -10 :sigma 1}}}
             {:probability 0.25
              :parameters  {:x {"yes" 0.0 "no" 1.0}
                            :y {:mu 10 :sigma 1}}}]]}))

(defn almost-equal?
  ([x y]
   (almost-equal? x y 10E-4))
  ([x y threshold]
   (< (Math/abs (- x y))
      threshold)))

(deftest approximate-equality
  (are [x y] (almost-equal? x y)
    10E-4 10E-5
    10E-5 10E-4)
  (are [x y] (not (almost-equal? x y))
    10E-3 10E-4
    10E-4 10E-3))

(deftest constrain
  (let [sexp-opts {:operation? seq?
                   :variable? symbol?
                   :operands rest
                   :operator first}
        gpm (reify gpm.proto/GPM
              (logpdf [_ target constraints]
                0.5)
              (simulate [_ target constraints]
                {'x (rand-nth [true false])}))
        cgpm (constrained/constrain gpm '(= x true) sexp-opts)]
    (is (every? #{'{x true}} (repeatedly 100 #(gpm/simulate cgpm '[x] {})))))
  (testing "multimixture"
    (let [opts {:operation? seq?
                :variable? keyword?
                :operands rest
                :operator first}]
      (testing "categorical target"
        (let [cgpm (constrained/constrain mmix '(> :y 0) opts)]
          (testing "simulate"
            (is (every? #{{:x "no"}}   (repeatedly 100 #(gpm/simulate cgpm [:x] {}))))
            (is (every? (comp pos? :y) (repeatedly 100 #(gpm/simulate cgpm [:y] {})))))
          (testing "logpdf"
            (is (=             1.0 (Math/exp (gpm/logpdf cgpm {:x "no"}  {}))))
            (is (almost-equal? 0.0 (Math/exp (gpm/logpdf cgpm {:x "yes"} {})))))))
      (testing "numerical target"
        (let [cgpm (constrained/constrain mmix '(= :x "no") opts)]
          (testing "simulate"
            (let [average-y (transduce (map :y)
                                       rfs/avg
                                       (repeatedly 100 #(gpm/simulate cgpm [:y] {})))]
              (is (almost-equal? 10 average-y 0.5))))
          (testing "logpdf"
            (let [{:keys [mu sigma]} (get-in mmix [:views 0 1 :parameters :y])]
              (is (almost-equal? (Math/exp (distributions/score-gaussian 9 [mu sigma]))
                                 (Math/exp (gpm/logpdf cgpm {:y 9} {})))))))))))

(defn seq->next
  [coll]
  (let [coll (atom (cycle coll))]
    (fn []
      (let [x (first @coll)]
          (swap! coll rest)
          x))))

(defn seq->simulator
  [coll]
  (let [next (seq->next coll)]
    (reify gpm.proto/GPM
      (logpdf [_ _ _]
        nil)
      (simulate [_ _ _]
        (next)))))

(deftest rejection
  (let [gpm (seq->simulator [{:x -2 :y -1}
                             {:x  2 :y  1}])
        cgpm (constrained/->ConstrainedGPM gpm #(pos? (:y %)) #{:x} 1000)]
    (is (= {:x 2} (gpm/simulate cgpm [:x] {})))))

(deftest mean
  (let [next (seq->next  [(Math/log 1) (Math/log 0)])
        gpm (reify gpm.proto/GPM
              (logpdf [_ _ _]
                (next))
              (simulate [_ _ _]
                nil))
        cgpm (constrained/->ConstrainedGPM gpm (constantly true) #{:x} 10)]
    (is (almost-equal? 0.5 (Math/exp (gpm/logpdf cgpm {} {}))))))

(deftest condition-categorical
  (let [opts {:operation? seq?
              :variable? keyword?
              :operands rest
              :operator first}
        categorical (categorical/spec->categorical
                     :x
                     :suff-stats {:n 100 :counts {"a" 10 "b" 10 "c" 80}}
                     :hyperparameters {:alpha 0})
        gpm (reify gpm.proto/GPM
              (logpdf [_ targets constraints]
                (gpm/logpdf categorical targets constraints))
              (simulate [_ targets constraints]
                {:x (gpm/simulate categorical targets constraints)}))
        event '(or (= :x "a")
                   (= :x "b"))
        cgpm (constrained/constrain gpm event opts)]
    (is (almost-equal? 0.5 (Math/exp (gpm/logpdf cgpm {:x "a"} {})) 10E-2))
    (is (almost-equal? 0.5 (Math/exp (gpm/logpdf cgpm {:x "b"} {})) 10E-2))
    (is (almost-equal? 0.0 (Math/exp (gpm/logpdf cgpm {:x "c"} {})) 10E-2))))

(defspec delegation
  (testing "variables"
    (prop/for-all [vs (gen/vector gen/keyword)]
      (let [model (reify gpm.proto/Variables
                    (variables [_]
                      vs))
            constrained-model (constrained/constrain
                               model
                               '(= x 0)
                               {:operation? seq?
                                :operator first
                                :operands rest
                                :variable? symbol?})]
        (=  (gpm/variables model)
            (gpm/variables constrained-model))))))
