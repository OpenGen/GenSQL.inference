(ns gensql.inference.gpm.column-test
  (:require [gensql.inference.gpm.column :as column]
            [clojure.math :as math]
            [clojure.test :as test :refer [are deftest is]]
            [gensql.inference.utils :as utils]
            [gensql.inference.gpm :as gpm]
            [gensql.inference.gpm.proto :as gpm.proto]))

(def data-bernoulli
  [true true true true false false])

(def data-categorical
  ["red" "red" "red" "red" "blue" "green"])

(def data-gaussian
  [6 6 6 4 4 4])

(def latents
  {:alpha 1
   :counts {:one 4 :two 2}
   :y {0 :one
       1 :one
       2 :one
       3 :one
       4 :two
       5 :two}})

(def hypers-bernoulli
  {:alpha 1 :beta 1})

(def hypers-categorical
  {:alpha 2})

(def hypers-gaussian
  {:m 0 :r 1 :s 2 :nu 3})

(def column-bernoulli
  (column/construct-column-from-latents "flip"
                                        :bernoulli
                                        hypers-bernoulli
                                        latents
                                        (into {} (map-indexed vector data-bernoulli))))

(def column-categorical
  (column/construct-column-from-latents "color"
                                        :categorical
                                        hypers-categorical
                                        latents
                                        (into {} (map-indexed vector data-categorical))
                                        {:options {"color" ["red" "blue" "green"]}}))

(def column-gaussian
  (column/construct-column-from-latents "height"
                                        :gaussian
                                        hypers-gaussian
                                        latents
                                        (into {} (map-indexed vector data-gaussian))))

;; Verifies that creating a column with given latent assignments is deterministic.
(deftest create-column-smoke-test
  (is (= 1 (->> #(column/construct-column-from-latents "flip"
                                                               :bernoulli
                                                               hypers-bernoulli
                                                               latents
                                                               (zipmap (range) data-bernoulli))
                (repeatedly 1000)
                (distinct)
                (count)))))

(defn absolute-difference
  "Calculates absolute value of the difference of a and b."
  [a b]
  (abs (- a b)))

;; Checks logpdf across the different primitive types.
(deftest logpdf
  (let [;; alpha' = alpha + # true, beta' = beta + # false
        ;; alpha-1 = 1 + 4 = 5, beta-1 = 1 + 0 = 1
        ;; alpha-2 = 1 + 0 = 1, beta-2 = 1 + 2 = 3
        ;; logP(true | column-bernoulli) = logsumexp(weight-1 + logP(true | category-1),
        ;;                                           weight-2 + logP(true | category-2),
        ;;                                           weight-aux + logP(true | category-aux))
        ;;                               = logsumexp(ln(4/7) + ln(5/6), ln(2/7) + ln(1/4), ln(1/7) + ln(1/2))
        ;;                               = -0.4795730803
        bernoulli-true-sol -0.4795730803
        bernoulli-true (gpm.proto/logpdf column-bernoulli {(:var-name column-bernoulli) true} {})
        ;; logP(false | column-bernoulli) = logsumexp(weight-1 + logP(false | category-1),
        ;;                                            weight-2 + logP(false | category-2))
        ;;                                = logsumexp(ln(4/7) + ln(1/6), ln(2/7) + ln(3/4), ln(1/7) + ln(1/2))
        ;;                                = -0.965080896
        bernoulli-false-sol -0.965080896
        bernoulli-false (gpm.proto/logpdf column-bernoulli {(:var-name column-bernoulli) false} {})

        ;; alpha comes from the CRP.
        ;; alpha-c is the hyperparameter for a symmetric Dirichlet prior.
        ;; logP("red" | column-categorical) = logsumexp(weight-1 + logP("red" | category-1),
        ;;                                              weight-2 + logP("red" | category-2),
        ;;                                              weight-aux + logP("red" | category-aux)
        ;;                                  = logsumexp(ln(4/7) + ln(6/10), ln(2/7) + ln(2/8), ln(1/7) + ln(1/3))
        ;;                                  = -0.7723965522
        categorical-red-sol -0.7723965522
        categorical-red (gpm.proto/logpdf column-categorical {(:var-name column-categorical) "red"} {})

        ;; logP("blue" | column-categorical) = logsumexp(weight-1 + logP("blue" | category-1),
        ;;                                               weight-2 + logP("blue" | category-2),
        ;;                                               weight-aux + logP("blue" | category-aux)
        ;;                                  = logsumexp(ln(4/7) + ln(2/10), ln(2/7) + ln(3/8), ln(1/7) + ln(1/3))
        ;;                                  = -1.3128668926
        categorical-blue-sol -1.3128668926
        categorical-blue (gpm.proto/logpdf column-categorical {(:var-name column-categorical) "blue"} {})
        categorical-green (gpm.proto/logpdf column-categorical {(:var-name column-categorical) "green"} {})

        threshold 1e-5]
    (is (utils/almost-equal? bernoulli-true-sol bernoulli-true absolute-difference threshold))
    (is (utils/almost-equal? bernoulli-false-sol bernoulli-false absolute-difference threshold))
    (is (utils/almost-equal? 1
                             (+ (math/exp bernoulli-true) (math/exp bernoulli-false))
                             absolute-difference
                             threshold))

    (is (utils/almost-equal? categorical-red-sol categorical-red absolute-difference threshold))
    (is (utils/almost-equal? categorical-blue-sol categorical-blue absolute-difference threshold))
    (is (utils/almost-equal? 1
                             (+ (math/exp categorical-red) (math/exp categorical-blue) (math/exp categorical-green))
                             absolute-difference
                             threshold))))

;; Checks logpdf across the different primitive types.
(deftest simulate
  (let [n-samples-bernoulli 1000
        threshold 0.1
        ;; alpha' = alpha + # true, beta' = beta + # false
        ;; alpha-1 = 1 + 4 = 5, beta-1 = 1 + 0 = 1
        ;; alpha-2 = 1 + 0 = 1, beta-2 = 1 + 2 = 3
        ;; mean-1 = alpha-1/(alpha-1 + beta-1) = 5/6
        ;; mean-2 = alpha-2/(alpha-2 + beta-2) = 1/4
        ;; mean-aux = alpha/(alpha + beta) = 1/2
        ;; bernoulli-mean = weight-1 * mean-1 + weight-2 * mean-2 + weight-aux * mean-aux
        ;;                = 4/7 * 5/6 + 2/7 * 1/4 + 1/7 * 1/2
        ;;                = 0.619047619
        bernoulli-mean 0.619047619
        bernoulli-emp-mean (double (utils/average (map (fn [sample] (if sample 1 0))
                                                       (repeatedly n-samples-bernoulli
                                                                   #(gpm.proto/simulate column-bernoulli ["flip"] {})))))

        n-samples-categorical 1000
        ;; alpha-red-mean = 4/7 * 6/10 + 2/7 * 2/8 + 1/7 * 1/3 = 0.4619047619
        ;; alpha-blue-mean = 4/7 * 2/10 + 2/7 * 3/8 + 1/7 * 1/3 = 0.269047619
        ;; alpha-green-mean = 4/7 * 2/10 + 2/7 * 3/8 + 1/7 * 1/3 = 0.269047619
        ;; Then we normalize and create the empirical distribution.
        categorical-dist {"red" 0.4619047619 "blue" 0.269047619 "green" 0.269047619}
        categorical-emp-dist (reduce-kv (fn [m k v]
                                          (assoc m k (double (/ v n-samples-categorical))))
                                        {}
                                        (frequencies (repeatedly n-samples-categorical #(gpm.proto/simulate column-categorical ["color"] {}))))

        n-samples-gaussian 1000
        gaussian-threshold 0.5
        ;; mu-1 = (r * m + sum-x-1) / (r + n) = (1 * 0 + 22)/ (1 + 4) = 4.4
        ;; mu-2 = (r * m + sum-x-2) / (r + n) = (1 * 0 + 8)/ (1 + 2) = 2.6666666667
        ;; mu-aux = (r * m) / r = 0
        ;; mean-mu = 4/7 * 4.4 + 2/7 * 2.6666666667 + 1/7 * 0 = 3.2761904762
        gaussian-mean 3.2761904762
        gaussian-emp-mean (double (utils/average (repeatedly n-samples-gaussian #(gpm.proto/simulate column-gaussian ["height"] {}))))]
    (is (utils/almost-equal? bernoulli-emp-mean bernoulli-mean absolute-difference threshold))
    (is (utils/almost-equal-maps? categorical-dist categorical-emp-dist absolute-difference threshold))
    (is (utils/almost-equal? gaussian-emp-mean gaussian-mean absolute-difference gaussian-threshold))))

(deftest variables
  (are [variable gpm] (= #{variable} (gpm/variables gpm))
    "flip" column-bernoulli
    "color" column-categorical
    "height" column-gaussian))
