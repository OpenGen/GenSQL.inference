(ns inferenceql.inference.gpm.primitive-gpms.gaussian-test
  (:require [clojure.math :as math]
            [clojure.test :as test :refer [deftest is]]
            [inferenceql.inference.gpm :as gpm]
            [inferenceql.inference.gpm.primitive-gpms.gaussian :as gaussian]
            [inferenceql.inference.gpm.proto :as gpm.proto]
            [inferenceql.inference.utils :as utils]))

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

#?(:clj (deftest logprob
  (let [a 1
        b 2
        c 3
        s1 '<
        s2 '>
        hyperparameters {:m 0 :r 1 :s 1 :nu 1}
        m    (:m hyperparameters)
        r    (:r hyperparameters)
        s    (:s hyperparameters)
        nu   (:nu hyperparameters)
        suff-stats {:n 0 :sum-x 0 :sum-x-sq 0}
        n        (:n suff-stats)
        sum-x    (:sum-x suff-stats)
        sum-x-sq (:sum-x-sq suff-stats)
        ; Manuall converting parametes, following https://github.com/probcomp/cgpm/blob/master/tests/test_teh_murphy.py
        ; for the conversion of the hyperparameters into the parameters for a Student t
        ; distribution (see also: https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf)
        rn  (+ r n)
        nun (+ nu n)
        loc  (/(+ (* r m) sum-x) rn)
        sn  (+ s sum-x-sq (* r m m) (* -1 rn loc loc))
        an  (/ nun 2)
        degrees-of-freedom  (* 2 an)
        bn  (/ sn 2)
        scale (math/sqrt (/ (* bn (+ rn 1)) (* an rn)))]
    (is (= (math/log (gaussian/student-t-cdf a degrees-of-freedom loc scale))
           (gpm/logprob gaussian-pgpm [s1 (symbol "x") a])))
    (is (= (math/log (gaussian/student-t-cdf b degrees-of-freedom loc scale))
           (gpm/logprob gaussian-pgpm [s1 (symbol "x") b])))
    (is (= (math/log (gaussian/student-t-cdf c degrees-of-freedom loc scale))
           (gpm/logprob gaussian-pgpm [s1 (symbol "x") c])))
    (is (= (math/log (- 1 (gaussian/student-t-cdf a degrees-of-freedom loc scale)))
           (gpm/logprob gaussian-pgpm [s2 (symbol "x") a])))
    (is (= (math/log (- 1 (gaussian/student-t-cdf b degrees-of-freedom loc scale)))
           (gpm/logprob gaussian-pgpm [s2 (symbol "x") b])))
    (is (= (math/log (- 1 (gaussian/student-t-cdf c degrees-of-freedom loc scale)))
           (gpm/logprob gaussian-pgpm [s2 (symbol "x") c])))
    (is (= (math/log (gaussian/student-t-cdf a degrees-of-freedom loc scale))
           (gpm/logprob gaussian-pgpm ['not [s2 (symbol "x") a]]))))))
