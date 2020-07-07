(ns inferenceql.inference.gpm.primitive-gpms.gaussian-test
  (:require [clojure.test :as test :refer [deftest is]]
            [inferenceql.inference.gpm.proto :as gpm.proto]
            [inferenceql.inference.utils :as utils]
            [inferenceql.inference.gpm.primitive-gpms.gaussian :as gaussian]))

(def gaussian-pgpm
  (let [var-name "gaussian"
        suff-stats {:n 0 :sum-x 0 :sum-x-sq 0}]
    (gaussian/spec->gaussian var-name :suff-stats suff-stats)))

(deftest logpdf
  (let [targets {"gaussian" 0}
        constraints {"gaussian" 1}]
    ;; See http://www.stats.ox.ac.uk/~teh/research/notes/GaussianInverseGamma.pdf for
    ;; calculation.
    (is (utils/almost-equal? (Math/log 0.22507) (gpm.proto/logpdf gaussian-pgpm targets {}) utils/relerr 1e-3))
    (is (= 1.0 (Math/exp (gpm.proto/logpdf gaussian-pgpm {} {}))))
    (is (= 1.0 (Math/exp (gpm.proto/logpdf gaussian-pgpm targets targets))))
    (is (= ##-Inf (gpm.proto/logpdf gaussian-pgpm targets constraints)))))

(deftest simulate
  (let [n 100000
        error-margin 20
        targets []
        constraints {}
        average (/ (reduce + (gpm.proto/simulate gaussian-pgpm targets constraints n))
                   n)]
    (is (< (utils/abs average)
           error-margin))))

(defn check-suff-stats
  [stattype suff-stats]
  (is (= (:suff-stats stattype) suff-stats))
  stattype)

(def data
  {0 1
   1 1
   2 2
   3 3})

(deftest incorporate-unincorporate
  (let [suff-stats-0 {:n 0 :sum-x 0 :sum-x-sq 0}
        suff-stats-1 {:n 1 :sum-x 1 :sum-x-sq 1}
        suff-stats-2 {:n 2 :sum-x 3 :sum-x-sq 5}
        suff-stats-3 {:n 1 :sum-x 1 :sum-x-sq 1}]
    (-> gaussian-pgpm
        (check-suff-stats suff-stats-0)
        (gpm.proto/incorporate {(:var-name gaussian-pgpm) (get data 1)})
        (check-suff-stats suff-stats-1)
        (gpm.proto/incorporate {(:var-name gaussian-pgpm) (get data 2)})
        (check-suff-stats suff-stats-2)
        (gpm.proto/unincorporate {(:var-name gaussian-pgpm) (get data 2)})
        (check-suff-stats suff-stats-3))))

(deftest logpdf-score
  (let [one-observation (gpm.proto/incorporate gaussian-pgpm {(:var-name gaussian-pgpm) (get data 0)})
        two-observations (-> gaussian-pgpm
                             (gpm.proto/incorporate {(:var-name gaussian-pgpm) (get data 0)})
                             (gpm.proto/incorporate {(:var-name gaussian-pgpm) (get data 1)}))
        score-one (gpm.proto/logpdf-score one-observation)
        score-two (gpm.proto/logpdf-score two-observations)]
    (is (> score-one score-two))))

(deftest gaussian?
  (is (gaussian/gaussian? gaussian-pgpm))
  (is (not (gaussian/gaussian? {:p 0.5}))))

(deftest spec->gaussian
  (let [var-name "gaussian"
        gaussian (gaussian/spec->gaussian var-name)]
    (is (gaussian/gaussian? gaussian))
    (is (gaussian/gaussian? (gaussian/spec->gaussian var-name :suff-stats {:n 0 :sum-x 0 :sum-x-sq 0})))
    (is (gaussian/gaussian? (gaussian/spec->gaussian var-name :suff-stats {:n 0 :sum-x 0 :sum-x-sq 0} :hyperparameters{:m 0 :r 1 :s 1 :nu 1})))))
