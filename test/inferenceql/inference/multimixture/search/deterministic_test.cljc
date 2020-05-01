(ns inferenceql.inference.multimixture.search.deterministic-test
  (:require [clojure.spec.alpha :as s]
            [clojure.test :as test :refer [deftest testing is]]
            [clojure.walk :as walk :refer [stringify-keys]]
            #?(:clj [clojure.string :as str])
            [inferenceql.inference.utils :as utils]
            [inferenceql.inference.multimixture.search.deterministic :as deterministic]
            [metaprob.distributions :as dist]))

(def two-gaussian-no-observations-spec
  {:vars {"x" :gaussian}
   :views [[{:probability 0.5
             :parameters {"x" {:mu 0 :sigma 1}}}
            {:probability 0.5
             :parameters {"x" {:mu 5 :sigma 1}}}]]})

(deftest two-components-no-observation
  (let [known-rows []
        unknown-rows [{"x" 0} {"x" 5}]
        beta-params {:alpha 0.001 :beta 0.001}
        results (deterministic/search two-gaussian-no-observations-spec
                                      "y"
                                      known-rows
                                      unknown-rows
                                      beta-params)]
    (is (= [0.5 0.5] results))))
