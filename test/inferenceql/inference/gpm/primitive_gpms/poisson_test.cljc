(ns inferenceql.inference.gpm.primitive-gpms.poisson-test
  (:require [clojure.math :as math]
            [clojure.test :as test :refer [deftest is]]
            [inferenceql.inference.gpm :as gpm]
            [inferenceql.inference.gpm.primitive-gpms.poisson :as poisson]
            [inferenceql.inference.gpm.proto :as gpm.proto]
            [inferenceql.inference.utils :as utils]))

(def var-name "poisson")

(def poisson-pgpm
  (let [suff-stats {:n 0 :sum-x 0 :sum-log-fact 0}
        hypers {:a 2 :b 2}]
    (poisson/spec->poisson var-name :suff-stats suff-stats :hyperparameters hypers)))

(deftest logpdf
  (let [targets {"poisson" 0}
        constraints {"poisson" 1}]
    (is (= 1.0 (math/exp (gpm.proto/logpdf poisson-pgpm {} {}))))
    (is (= 1.0 (math/exp (gpm.proto/logpdf poisson-pgpm targets targets))))
    (is (= ##-Inf (gpm.proto/logpdf poisson-pgpm targets constraints))))
    (is (utils/almost-equal? -1.2163953243244932 (gpm.proto/logpdf poisson-pgpm {"poisson" 1} {}) utils/relerr 1E-8)))
