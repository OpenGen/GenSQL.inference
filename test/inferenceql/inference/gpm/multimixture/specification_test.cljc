(ns inferenceql.inference.gpm.multimixture.specification-test
  (:require [clojure.spec.alpha :as s]
            [clojure.test :as test :refer [deftest testing is]]
            [expound.alpha :as expound]
            [inferenceql.inference.gpm.multimixture.specification :as spec]))

(def mmix
  {:vars {"x" :gaussian
          "y" :categorical
          "a" :gaussian
          "b" :categorical}
   :views [[{:probability 1
             :parameters {"x" {:mu 2 :sigma 3}
                          "y" {"0" 0.1 "1" 0.2 "2" 0.3 "3" 0.4}}}]
           [{:probability 0.4
             :parameters {"a" {:mu 4 :sigma 5}
                          "b" {"0" 0.9, "1" 0.01, "2" 0.02, "3" 0.03, "4" 0.04}}}
            {:probability 0.6
             :parameters {"a" {:mu 6 :sigma 7}
                          "b" {"0" 0.99, "1" 0.001, "2" 0.002, "3" 0.003, "4" 0.004}}}]]})

(deftest mmix-is-valid
  (when-not (s/valid? ::spec/multi-mixture mmix)
    (expound/expound ::spec/multi-mixture mmix))
  (is (s/valid? ::spec/multi-mixture mmix)))

(deftest view-variables-test
  (is (= #{"x" "y"} (spec/view-variables (first (:views mmix)))))
  (is (= #{"a" "b"} (spec/view-variables (second (:views mmix))))))

(deftest variables-test
  (is (= #{"x" "y" "a" "b"} (spec/variables mmix))))

(deftest stattype-test
  (is (= :gaussian (spec/stattype mmix "x"))))

(deftest statistical-types-test
  (let [variables (spec/variables mmix)]
    (testing "numerical?"
      (is (= #{"x" "a"}
             (into #{}
                   (filter #(spec/numerical? mmix %))
                   variables))))
    (testing "nominal?"
      (is (= #{"y" "b"}
             (into #{}
                   (filter #(spec/nominal? mmix %))
                   variables))))))

(deftest parameters-test
  (is (= {:mu 2 :sigma 3} (spec/parameters mmix "x" 0))))

(deftest categories-test
  (is (= #{"0" "1" "2" "3"}     (spec/categories mmix "y")))
  (is (= #{"0" "1" "2" "3" "4"} (spec/categories mmix "b"))))

(deftest categorical-probabilities-test
  (is (= {"0" 0.1 "1" 0.2 "2" 0.3 "3" 0.4} (spec/categorical-probabilities mmix "y" 0)))
  (is (= {"0" 0.5 "1" 0.5}
         (let [mmix {:vars {"x" :categorical}
                     :views [[{:probability 0.5
                               :parameters {"x" {"0" 0.4 "1" 0.6}}}
                              {:probability 0.5
                               :parameters {"x" {"0" 0.6 "1" 0.4}}}]]}]
           (spec/categorical-probabilities mmix "x" 0 1)))))
