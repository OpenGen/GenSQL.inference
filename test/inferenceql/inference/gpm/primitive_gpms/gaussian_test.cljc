(ns inferenceql.inference.gpm.primitive-gpms.gaussian-test
  (:require [clojure.math :as math]
            [clojure.test :as test :refer [deftest is]]
            [inferenceql.inference.gpm :as gpm]
            [inferenceql.inference.gpm.primitive-gpms.gaussian :as gaussian]
            [inferenceql.inference.gpm.proto :as gpm.proto]
            [inferenceql.inference.utils :as utils]
            [fastmath.random :as r]))

(def var-name "gaussian")

(def gaussian-pgpm
  (let [suff-stats {:n 0 :sum-x 0 :sum-x-sq 0}]
    (gaussian/spec->gaussian var-name :suff-stats suff-stats)))

(deftest logpdf
  (let [targets {"gaussian" 0}
        constraints {"gaussian" 1}]
    ;; See http://www.stats.ox.ac.uk/~teh/research/notes/GaussianInverseGamma.pdf for
    ;; calculation.
    (is (utils/almost-equal? (math/log 0.22507) (gpm.proto/logpdf gaussian-pgpm targets {}) utils/relerr 1e-3))
    (is (= 1.0 (math/exp (gpm.proto/logpdf gaussian-pgpm {} {}))))
    (is (= 1.0 (math/exp (gpm.proto/logpdf gaussian-pgpm targets targets))))
    (is (= ##-Inf (gpm.proto/logpdf gaussian-pgpm targets constraints)))))


(deftest simulate
  (let [n 10000
        targets []
        constraints {}
        incorporated-value 5
        gaussian-pgpm-simulate (-> gaussian-pgpm
                                   (gpm.proto/incorporate {var-name incorporated-value})
                                   (gpm.proto/incorporate {var-name incorporated-value})
                                   (gpm.proto/incorporate {var-name incorporated-value})
                                   (gpm.proto/incorporate {var-name incorporated-value})
                                   (gpm.proto/incorporate {var-name incorporated-value}))
        average (/ (reduce + (repeatedly n #(gpm.proto/simulate gaussian-pgpm-simulate targets constraints)))
                   n)]
    (is (< 4 average 5))))

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

(deftest variables
  (is (= #{var-name} (gpm/variables gaussian-pgpm))))

(deftest logprob
  (let [a 1
        b 2
        c 3
        s1 <
        s2 >
        hyperparameters {:m 0 :r 1 :s 1 :nu 1}
        m    (:m hyperparameters)
        r    (:r hyperparameters)
        s    (:s hyperparameters)
        nu   (:nu hyperparameters)
        suff-stats {:n 0 :sum-x 0 :sum-x-sq 0}
        n        (:n suff-stats)
        sum-x    (:sum-x suff-stats)
        sum-x-sq (:sum-x-sq suff-stats)
        rn  (+ r n)
        nun (+ nu n)
        mn  (/(+ (* r m) sum-x) rn)
        sn  (+ s sum-x-sq (* r m m) (* -1 rn mn mn))
        an  (/ nun 2)
        bn  (/ sn 2)
        scalesq (/ (* bn (+ rn 1)) (* an rn))
        params {:degrees-of-freedom an :loc mn :scale (math/sqrt scalesq)}]
  (is (= (math/log (r/cdf (r/distribution :t params) a ))
         (gpm/logprob gaussian-pgpm [s1 (symbol "x") a])))
  (is (= (math/log (r/cdf (r/distribution :t params) b))
         (gpm/logprob gaussian-pgpm [s1 (symbol "x") b])))
  (is (= (math/log (r/cdf (r/distribution :t params) c))
         (gpm/logprob gaussian-pgpm [s1 (symbol "x") c])))
  (is (= (math/log (- 1 (r/cdf (r/distribution :t params) a)))
         (gpm/logprob gaussian-pgpm [s2 (symbol "x") a])))
  (is (= (math/log (- 1 (r/cdf (r/distribution :t params) b)))
         (gpm/logprob gaussian-pgpm [s2 (symbol "x") b])))
  (is (= (math/log (- 1 (r/cdf (r/distribution :t params) c)))
         (gpm/logprob gaussian-pgpm [s2 (symbol "x") c])))
  (is (thrown? Exception (gpm/logprob gaussian-pgpm [(symbol "AND")
                                                     [> (symbol "x") 1]
                                                     [> (symbol "x") 1]])))))
;(logprob)
