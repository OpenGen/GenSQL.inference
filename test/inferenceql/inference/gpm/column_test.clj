(ns inferenceql.inference.gpm.column-test
  (:require [inferenceql.inference.gpm.column :as column]
            [clojure.test :as test :refer [deftest is]]
            [inferenceql.inference.utils :as utils]
            [inferenceql.inference.gpm.proto :as gpm.proto]))

(def data-bernoulli
  [true false true true true false])

(def data-categorical
  ["red" "blue" "green" "red" "blue" "red"])

(def data-gaussian
  [9 1 1 2 3 1])

(def latents
  {:alpha 1
   :counts {:one 5 :two 1}
   :y {0 :one
       1 :two
       2 :one
       3 :one
       4 :one
       5 :one}})

(def hypers-bernoulli
  {:alpha 2 :beta 0.9})

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
  (utils/abs (- a b)))

;; Checks logpdf across the different primitive types.
(deftest logpdf
  (let [;; alpha' = alpha + # true, beta' = beta + # false
        ;; alpha-1 = 2 + 4 = 6, beta-1 = 0.9 + 1 = 1.9
        ;; alpha-2 = 2 + 0 = 2, beta-2 = 0.9 + 1 = 1.9
        ;; logP(true | column-bernoulli) = logsumexp(weight-1 + logP(true | category-1),
        ;;                                           weight-2 + logP(true | category-2),
        ;;                                           weight-aux + logP(true | category-aux))
        ;;                               = logsumexp(ln(5/(6 + alpha)) + ln(6/7.9), ln(1/(6 + alpha)) + ln(2/3.9),
        ;;                                           ln(alpha/(6 + alpha)) + ln(2/2.9))
        ;;                               = -0.336483
        bernoulli-true-sol -0.336483
        bernoulli-true (gpm.proto/logpdf column-bernoulli {(:var-name column-bernoulli) true} {})
        ;; logP(false | column-bernoulli) = logsumexp(weight-1 + logP(false | category-1),
        ;;                                            weight-2 + logP(false | category-2))
        ;;                                = logsumexp(ln(5/(6 + alpha)) + ln(1.9/7.9), ln(1/(6 + alpha)) + ln(1.9/3.9),
        ;;                                            ln(alpha/(6 + alpha)) + ln(0.9/2.9))
        ;;                                = -1.25273
        bernoulli-false-sol -1.25273
        bernoulli-false (gpm.proto/logpdf column-bernoulli {(:var-name column-bernoulli) false} {})

        ;; alpha comes from the CRP.
        ;; alpha-c is the hyperparameter for a symmetric Dirichlet prior.
        ;; logP("red" | column-categorical) = logsumexp(weight-1 + logP("red" | category-1),
        ;;                                              weight-2 + logP("red" | category-2),
        ;;                                              weight-aux + logP("red" | category-aux)
        ;;                                  = logsumexp(ln(5/(6 + alpha)) + ln((alpha-c + 3)/(3*alpha-c + 5)),
        ;;                                              ln(1/(6 + alpha)) + ln(alpha-c/(3*alpha-c + 1)),
        ;;                                              ln(1/(6 + alpha)) + ln(alpha-c/(3*alpha-c)))
        ;;                                  = -0.88404
        categorical-red-sol -0.88404
        categorical-red (gpm.proto/logpdf column-categorical {(:var-name column-categorical) "red"} {})

        ;; logP("blue" | column-categorical) = logsumexp(weight-1 + logP("blue" | category-1),
        ;;                                               weight-2 + logP("blue" | category-2),
        ;;                                               weight-aux + logP("blue" | category-aux)
        ;;                                  = logsumexp(ln(5/(6 + alpha)) + ln((alpha-c + 1)/(3*alpha-c + 5)),
        ;;                                              ln(1/(6 + alpha)) + ln((alpha-c + 1)/(3*alpha-c + 1)),
        ;;                                              ln(alpha/(6 + alpha)) + ln(alpha-c/(3*alpha-c)))
        ;;                                  = -1.19188
        categorical-blue-sol -1.19188
        categorical-blue (gpm.proto/logpdf column-categorical {(:var-name column-categorical) "blue"} {})
        categorical-green (gpm.proto/logpdf column-categorical {(:var-name column-categorical) "green"} {})

        threshold 1e-5]
    (is (utils/almost-equal? bernoulli-true-sol bernoulli-true absolute-difference threshold))
    (is (utils/almost-equal? bernoulli-false-sol bernoulli-false absolute-difference threshold))
    (is (utils/almost-equal? 1
                             (+ (Math/exp bernoulli-true) (Math/exp bernoulli-false))
                             absolute-difference
                             threshold))

    (is (utils/almost-equal? categorical-red-sol categorical-red absolute-difference threshold))
    (is (utils/almost-equal? categorical-blue-sol categorical-blue absolute-difference threshold))
    (is (utils/almost-equal? 1
                             (+ (Math/exp categorical-red) (Math/exp categorical-blue) (Math/exp categorical-green))
                             absolute-difference
                             threshold))))
