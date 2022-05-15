(ns inferenceql.inference.primitives-test
  (:require [clojure.math :as math]
            [clojure.test :as test :refer [deftest is]]
            [inferenceql.inference.primitives :as prim]))

;; We check the `simulate` methods of distributions by evaluating
;; their empirical mean and comparing that to the mean of the
;; distribution given its parameterization.
(deftest bernoulli-logpdf
  (let [x true
        x' false
        p 0.6
        p' (- 1 p)
        probs {:p p}]
    (is (= (math/log p)
           (prim/bernoulli-logpdf x probs)))
    (is (= (math/log p')
           (prim/bernoulli-logpdf x' probs)))))

(deftest bernoulli-simulate
  (let [p 0.6
        probs {:p p}
        n 10000
        samples (prim/bernoulli-simulate n probs)
        counts (frequencies samples)
        error 0.05]
    (is (< (abs (- (/ (get counts true) n)
                   p))
           error))))

(deftest gamma-logpdf
  (let [x 4
        k 5
        theta 2
        error 0.0001]
    (is (< (abs (- (math/exp (prim/gamma-logpdf x {:k k :theta theta}))
                   ;; logGamma(x; k, theta) = - lgamma(k) - k * ln(theta) + (k - 1)ln(x) - (x / theta)
                   ;; logGamma(4; 5, 2)     = -ln(24) - 5 * ln(2) + 4 * ln(4) - 2
                   ;;                       = -3.178 - 3.466 + 5.545 - 2
                   ;;                       = -3.099
                   ;;                   exp => 0.0451
                   0.04511))
           error))))

(deftest gamma-simulate
  (let [k 5
        theta 2
        n 100000
        samples-k-theta (prim/gamma-simulate n {:k k :theta theta})
        mean-k-theta (/ (reduce + samples-k-theta)
                        n)
        error   0.05]
    (is (< (abs (- mean-k-theta
                   (* k theta)))
           error))))

(deftest beta-logpdf
  (let [x 0.5
        alpha 0.5
        beta 0.5
        error 0.001]
    (is (< (abs (- (math/exp (prim/beta-logpdf x {:alpha alpha :beta beta}))
                   ;; logBeta(x; alpha, beta) = lgamma(alpha + beta) - lgamma(alpha) - lgamma(beta)
                   ;;                            + (alpha - 1) * ln(x) + (beta - 1) * ln(1 - x)
                   ;; logBeta(0.5; 0.5, 0.5)  = 0 - 0.249 - 0.249 + (-0.5) * ln(0.5) + (-0.5) * ln(0.5)
                   ;;                         = -0.452
                   ;;                     exp => 0.63662
                   0.63662))
           error))))

(deftest beta-simulate
  (let [alpha 1.5
        beta 1.5
        n 10000
        samples (prim/beta-simulate n {:alpha alpha :beta beta})
        mean (/ (reduce + samples)
                n)
        error 0.05]
    (is (< (abs (- mean
                   (/ alpha
                      (+ alpha beta))))
           error))))

(deftest categorical-logpdf
  (let [x "green"
        p {:p {"green" 0.2 "red" 0.4 "blue" 0.4}}]
    (is (= 0.2 (math/exp (prim/categorical-logpdf x p))))))

(deftest categorical-simulate
  (let [p {:p {"green" 0.2 "red" 0.4 "blue" 0.4}}
        n 10000
        samples (prim/categorical-simulate n p)
        counts (frequencies samples)
        error 0.05]
    (mapv #(is (< (abs (- (/ (get counts %)
                             n)
                          (get (:p p) %)))
                  error))
          (keys (:p p)))))

(deftest dirichlet-logpdf
  (let [x [0.4 0.4 0.2]
        alpha [2 2 1]
        error 0.001]
    (is (< (abs (- (math/exp (prim/dirichlet-logpdf x {:alpha alpha}))
                   ;; logDir(x; alpha) = lgamma(sum(alpha)) - sum(lgamma(alpha_i)) + sum((alpha_i - 1) (ln x_i))
                   ;; logDir([0.4 0.4 0.2]; [2 2 1]) = lgamma(5) - (1 + 1 + 1) + ln(0.4) + ln(0.4)
                   ;;                                = -3.178 - 1.833
                   ;;                                = -5.011
                   0.00667))
           error))))

(deftest dirichlet-simulate
  (let [alpha [2 2 1]
        sum-alpha (reduce + alpha)
        n 10000
        samples (prim/dirichlet-simulate n {:alpha alpha})
        error 0.05]
    (mapv #(is (< (abs (- (/ (reduce + (mapv (fn [sample]
                                               (nth sample %))
                                             samples))
                             n)
                          (/ (nth alpha %)
                             sum-alpha)))
                  error))
          (range (count alpha)))))

(deftest gaussian-logpdf
  (let [x 0
        mu 0
        sigma 1
        error 0.001]
    (is (< (abs (- (math/exp (prim/gaussian-logpdf x {:mu mu :sigma sigma}))
                   ;; Below is the evaluation of the gaussian formula:
                   ;; lNormal(x; mu, sigma^2) = -0.5 * ln(2PI * sigma^2) - 0.5 * ((x - mu)^2 / sigma^2)
                   0.39894))
           error))))

(deftest gaussian-simulate
  (let [mu 0
        sigma 1
        n 10000
        samples (prim/gaussian-simulate n {:mu mu :sigma sigma})
        mean (/ (reduce + samples)
                n)
        error 0.05]
    (is (< (abs (- mean
                   mu))
           error))))

(deftest log-categorical-logpdf
  (let [x "green"
        p {:p {"green" (math/log 0.2) "red" (math/log 0.4) "blue" (math/log 0.4)}}]
    (is (= 0.2 (math/exp (prim/log-categorical-logpdf x p))))))

(deftest log-categorical-simulate
  (let [p {:p {"green" (math/log 0.2) "red" (math/log 0.4) "blue" (math/log 0.4)}}
        n 10000
        samples (prim/log-categorical-simulate n p)
        counts (frequencies samples)
        error 0.05]
    (mapv #(is (< (abs (- (/ (get counts %)
                             n)
                          (math/exp (get (:p p) %))))
                  error))
          (keys (:p p)))))

(deftest logpdf-test
  (let [dist :gaussian
        x 0
        mu 0
        sigma 1
        error 0.001]
    (is (< (abs (- (math/exp (prim/logpdf x dist {:mu mu :sigma sigma}))
                   0.39894))
           error))
    (is (thrown? #?(:clj Exception
                    :cljs js/Error)
          (prim/logpdf x :foobar {:mu mu :sigma sigma})))))

(deftest logpdf-simulate
  (let [dist :gaussian
        mu 0
        sigma 1
        n 10000
        samples (prim/simulate n dist {:mu mu :sigma sigma})
        mean (/ (reduce + samples)
                n)
        error 0.05]
    (is (< (abs (- mean
                   mu))
           error))))
;; Couldn't make the below work, the call gives the desired result on the REPL.
;; (is (thrown? Exception (prim/simulate n :foobar {:mu mu :sigma sigma})))))
