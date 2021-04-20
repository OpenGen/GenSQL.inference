(ns inferenceql.inference.gpm.conditioned-test
  (:require [clojure.test :refer [testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [inferenceql.inference.gpm :as gpm]
            [inferenceql.inference.gpm.proto :as gpm.proto]
            [inferenceql.inference.gpm.conditioned :as conditioned]))

(defspec delegation
  (testing "variables"
    (prop/for-all [vs (gen/vector gen/keyword)
                   conditions (gen/map gen/keyword gen/any)]
      (let [model (reify
                    gpm.proto/Variables
                    (variables [_]
                      vs))
            conditioned-model (conditioned/condition model conditions)]
        (=  (gpm/variables model)
            (gpm/variables conditioned-model))))))
