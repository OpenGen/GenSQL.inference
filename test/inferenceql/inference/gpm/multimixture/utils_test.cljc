(ns inferenceql.inference.gpm.multimixture.utils-test
  (:require [clojure.test :as test :refer [deftest is]]
            [metaprob.prelude :as mp]
            [inferenceql.inference.gpm.multimixture.utils :as mmix.utils]))

(deftest generator-parity
  (let [spec {:vars {"x" :gaussian
                     "y" :categorical
                     "a" :gaussian
                     "b" :categorical}
              :views [[{:probability 1
                        :parameters {"x" {:mu 2 :sigma 3}
                                     "y" {"0" 0.1 "1" 0.2 "2" 0.3 "3" 0.4}}}]
                      [{:probability 0.4
                        :parameters {"a" {:mu 4 :sigma 5}
                                     "b" {"0" 0.9 "1" 0.01 "2" 0.02 "3" 0.03 "4" 0.04}}}
                       {:probability 0.6
                        :parameters {"a" {:mu 6 :sigma 7}
                                     "b" {"0" 0.99 "1" 0.001 "2" 0.002 "3" 0.003 "4" 0.004}}}]]}
        generator (mmix.utils/row-generator spec)
        optimized-generator (mmix.utils/optimized-row-generator spec)
        [_ trace _] (mp/infer-and-score :procedure generator)
        [row0 _ _] (mp/infer-and-score :procedure generator :observation-trace trace)
        [row1 _ _] (mp/infer-and-score :procedure optimized-generator :observation-trace trace)]
    (is (= row0 row1))))
