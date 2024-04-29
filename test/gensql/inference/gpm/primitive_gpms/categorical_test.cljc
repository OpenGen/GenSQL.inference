(ns gensql.inference.gpm.primitive-gpms.categorical-test
  (:require [clojure.math :as math]
            [clojure.test :as test :refer [deftest is]]
            [gensql.inference.gpm :as gpm]
            [gensql.inference.gpm.primitive-gpms.categorical :as categorical]
            [gensql.inference.gpm.proto :as gpm.proto]
            [gensql.inference.utils :as utils]))

(def var-name "categorical")

(def categorical-pgpm
  (let [suff-stats {:n 0 :counts {"a" 0 "b" 0 "c" 0}}]
    (categorical/spec->categorical var-name :suff-stats suff-stats)))

(deftest logpdf
  (let [targets {"categorical" "a"}
        constraints {"categorical" "b"}
        threshold 1e-5]
    (is (utils/almost-equal? (double (/ 1 3)) (math/exp (gpm.proto/logpdf categorical-pgpm targets {})) utils/relerr threshold))
    (is (= 1.0 (math/exp (gpm.proto/logpdf categorical-pgpm {} {}))))
    (is (= 1.0 (math/exp (gpm.proto/logpdf categorical-pgpm targets targets))))
    (is (= ##-Inf (gpm.proto/logpdf categorical-pgpm targets constraints)))))

(deftest simulate
  (let [n 10000
        error-margin 0.1
        targets []
        constraints {}
        samples (frequencies (repeatedly n #(gpm.proto/simulate categorical-pgpm targets constraints)))]
    (is (every? identity (mapv (fn [k] (< (abs (- (/ (get samples k) n)
                                                        (/ 1 3)))
                                          error-margin))
                               (-> categorical-pgpm
                                   (get-in [:suff-stats :counts])
                                   (keys)))))))

(defn check-suff-stats
  [stattype suff-stats]
  (is (= (:suff-stats stattype) suff-stats))
  stattype)

(def data
  {0 "a"
   1 "b"
   2 "c"})

(deftest incorporate-unincorporate
  (let [suff-stats-0 {:n 0 :counts {"a" 0 "b" 0 "c" 0}}
        suff-stats-1 {:n 1 :counts {"a" 1 "b" 0 "c" 0}}
        suff-stats-2 {:n 2 :counts {"a" 1 "b" 1 "c" 0}}
        suff-stats-3 {:n 1 :counts {"a" 0 "b" 1 "c" 0}}]
    (-> categorical-pgpm
        (check-suff-stats suff-stats-0)
        (gpm.proto/incorporate {(:var-name categorical-pgpm) (get data 0)})
        (check-suff-stats suff-stats-1)
        (gpm.proto/incorporate {(:var-name categorical-pgpm) (get data 1)})
        (check-suff-stats suff-stats-2)
        (gpm.proto/unincorporate {(:var-name categorical-pgpm) (get data 0)})
        (check-suff-stats suff-stats-3))))

(deftest logpdf-score
  (let [one-observation (gpm.proto/incorporate categorical-pgpm {(:var-name categorical-pgpm) (get data 0)})
        two-observations (-> categorical-pgpm
                             (gpm.proto/incorporate {(:var-name categorical-pgpm) (get data 0)})
                             (gpm.proto/incorporate {(:var-name categorical-pgpm) (get data 1)}))
        score-one (gpm.proto/logpdf-score one-observation)
        score-two (gpm.proto/logpdf-score two-observations)]
    (is (> score-one score-two))))

(deftest categorical?
  (is (categorical/categorical? categorical-pgpm))
  (is (not (categorical/categorical? {:p {"a" (/ 1 3) "b" (/ 1 3) "c" (/ 1 3)}}))))

(deftest spec->categorical
  (let [var-name "categorical"
        categorical (categorical/spec->categorical var-name :options ["a" "b" "c"])]
    (is (categorical/categorical? categorical))
    (is (categorical/categorical? (categorical/spec->categorical var-name :suff-stats {:n 10 :counts {"a" 3 "b" 4 "c" 3}})))))

(deftest variables
  (is (= #{var-name} (gpm/variables categorical-pgpm))))

(deftest logprob
  (let [event ['= "categorical" "a"]
        threshold 1e-5]
    (is (utils/almost-equal? (double (/ 1 3)) (math/exp (gpm.proto/logprob categorical-pgpm event)) utils/relerr threshold))
    (is (utils/almost-equal? (double (/ 2 3)) (math/exp (gpm.proto/logprob categorical-pgpm ['not event])) utils/relerr threshold))))
