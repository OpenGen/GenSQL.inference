(ns gensql.inference.gpm.primitive-gpms.bernoulli-test
  (:require [clojure.math :as math]
            [clojure.test :as test :refer [deftest is]]
            [gensql.inference.gpm :as gpm]
            [gensql.inference.gpm.primitive-gpms.bernoulli :as bernoulli]
            [gensql.inference.gpm.proto :as gpm.proto]))

(def var-name "flip")

(def bernoulli-pgpm
  (let [suff-stats {:n 0 :x-sum 0}]
    (bernoulli/spec->bernoulli var-name :suff-stats suff-stats)))

(deftest logpdf
  (let [targets {"flip" true}
        constraints {"flip" false}]
    (is (= 0.5 (math/exp (gpm.proto/logpdf bernoulli-pgpm targets {}))))
    (is (= 1.0 (math/exp (gpm.proto/logpdf bernoulli-pgpm {} {}))))
    (is (= 1.0 (math/exp (gpm.proto/logpdf bernoulli-pgpm targets targets))))
    (is (= ##-Inf (gpm.proto/logpdf bernoulli-pgpm targets constraints)))))

(deftest simulate
  (let [n 100000
        error-margin 0.01
        targets []
        constraints {}
        samples (frequencies (repeatedly n #(gpm.proto/simulate bernoulli-pgpm targets constraints)))]
    (is (< (abs (- (/ (get samples true) n)
                         0.5))
           error-margin))))

(defn check-suff-stats
  [stattype suff-stats]
  (is (= (:suff-stats stattype) suff-stats))
  stattype)

(def data
  {0 true
   1 false
   2 false
   3 true})

(deftest incorporate-unincorporate
  (let [suff-stats-0 {:n 0 :x-sum 0}
        suff-stats-1 {:n 1 :x-sum 1}
        suff-stats-2 {:n 2 :x-sum 1}
        suff-stats-3 {:n 1 :x-sum 0}]
    (-> bernoulli-pgpm
        (check-suff-stats suff-stats-0)
        (gpm.proto/incorporate {(:var-name bernoulli-pgpm) (get data 0)})
        (check-suff-stats suff-stats-1)
        (gpm.proto/incorporate {(:var-name bernoulli-pgpm) (get data 1)})
        (check-suff-stats suff-stats-2)
        (gpm.proto/unincorporate {(:var-name bernoulli-pgpm) (get data 0)})
        (check-suff-stats suff-stats-3))))

(deftest logpdf-score
  (let [one-observation (gpm.proto/incorporate bernoulli-pgpm {(:var-name bernoulli-pgpm) (get data 0)})
        two-observations (-> bernoulli-pgpm
                             (gpm.proto/incorporate {(:var-name bernoulli-pgpm) (get data 0)})
                             (gpm.proto/incorporate {(:var-name bernoulli-pgpm) (get data 1)}))
        score-one (gpm.proto/logpdf-score one-observation)
        score-two (gpm.proto/logpdf-score two-observations)]
    (is (> score-one score-two))))

(deftest bernoulli?
  (is (bernoulli/bernoulli? bernoulli-pgpm))
  (is (not (bernoulli/bernoulli? {:p 0.5}))))

(deftest spec->bernoulli
  (is (bernoulli/bernoulli? (bernoulli/spec->bernoulli var-name)))
  (is (bernoulli/bernoulli? (bernoulli/spec->bernoulli var-name :suff-stats {:n 10 :x-sum 9})))
  (is (bernoulli/bernoulli? (bernoulli/spec->bernoulli var-name :suff-stats {:n 10 :x-sum 9} :hyperparameters {:alpha 0.6 :beta 0.4}))))

(deftest variables
  (is (= #{var-name} (gpm/variables bernoulli-pgpm))))
