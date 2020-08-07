(ns inferenceql.inference.gpm.view-test
  (:require [inferenceql.inference.gpm.view :as view]
            [clojure.test :as test :refer [deftest is]]
            [inferenceql.inference.utils :as utils]
            [inferenceql.inference.gpm.column :as column]
            [inferenceql.inference.gpm.proto :as gpm.proto]))

(def latents
  {:alpha 1
   :counts {:one 4 :two 2}
   :y {0 :one
       1 :one
       2 :one
       3 :one
       4 :two
       5 :two}})

(def data
  {0 {"color" "red" "height" 6 "flip" true}
   1 {"color" "red" "height" 6 "flip" true}
   2 {"color" "red" "height" 6 "flip" true}
   3 {"color" "red" "height" 4 "flip" true}
   4 {"color" "blue" "height" 4 "flip" false}
   5 {"color" "green" "height" 4 "flip" false}})

(def view-spec {:hypers {"color"  {:alpha 2}
                         "height" {:m 0 :r 1 :s 2 :nu 3}
                         "flip"  {:alpha 1 :beta 1}}})

(def types {"color"  :categorical
            "height" :gaussian
            "flip" :bernoulli})

(def options {"color" ["red" "blue" "green"]})

(def view-inf (view/construct-view-from-latents view-spec latents types data {:options options :crosscat true}))

;; Verifies that creating a View with given latent assignments is deterministic.
(deftest create-view-smoke-test
  (is (= 1 (->> #(view/construct-view-from-latents view-spec
                                                   latents
                                                   types
                                                   data
                                                   {:options options})
                (repeatedly 1000)
                (distinct)
                (count)))))

(defn absolute-difference
  "Calculates absolute value of the difference of a and b."
  [a b]
  (utils/abs (- a b)))

;; Checks logpdf across constrained and unconstrained queries.
(deftest logpdf
  (let [targets-unconstrained {"color" "red" "height" 6 "flip" true}
        empty-constraints {}
        targets-constrained {"color" "red" "height" 6}
        constraints {"flip" true}
        ;; We verify that calls to logpdf for a single column variable return the same results
        ;; as defined in the column tests for those primitive types and values.
        ;; Refer to `inferenceql.inference.gpm.column-test` for the mathematical explanation of the below.
        bernoulli-true-sol -0.4795730803
        categorical-red-sol -0.7723965522
        gaussian-6-sol -2.43534388877

        bernoulli-logp (gpm.proto/logpdf view-inf {"flip" true} empty-constraints)
        categorical-logp (gpm.proto/logpdf view-inf {"color" "red"} empty-constraints)
        gaussian-logp (gpm.proto/logpdf view-inf {"height" 6} empty-constraints)

        ;; We then consider an unconstrained query of all three variables.
        ;; Let w_i = weight of the ith category, denoted c_i. Then,
        ;; P(color = red, height = 6, flip = true | view) =
        ;;    w_0 * P(color = red, height = 6, flip = true | c_0)
        ;;    +  w_1 * P(color = red, height = 6, flip = true | c_1)
        ;; or, in the log space,
        ;; logP(color = red, height = 6, flip = true | view) =
        ;;    logsumexp(ln(w_0) + logP(color = red, height = 6, flip = true | c_0),
        ;;              ln(w_1) + logP(color = red, height = 6, flip = true | c_1))
        ;;              ln(w_aux) + logP(color = red, height = 6, flip = true | c_aux))
        ;;
        ;; We substitute values from the column logpdf tests:
        ;; logP(color = red, height = 6, flip = true | view) =
        ;;  = logsumexp(ln(4/7) + ln(6/10) + -2.041879 + ln(5/6),
        ;;              ln(2/7) + ln(2/8) + -3.0938743 + ln(1/4),
        ;;              ln(1/7) + ln(1/3) + -5.7499 + ln(1/2))
        ;;  = logsumexp(-3.2946419685, -7.1192259907, -9.4875696183)
        ;;  = -3.27105108673

        unconstrained-sol -3.27105108673
        unconstrained-logp (gpm.proto/logpdf view-inf targets-unconstrained empty-constraints)

        ;; Lastly, we consider a constrained query of two variables, conditioned on one other.
        ;; Remembering Bayes' rule,
        ;;                                             P(color = red, height = 6, flip = true)
        ;; P(color = red, height = 6 | flip = true) = -----------------------------------------
        ;;                                                         P(flip = true)
        ;; and the log yields
        ;; logP(color = red, height = 6 | flip = true) = logP(color = red, height = 6, flip = true)
        ;;                                                 - logP(flip = true)
        ;; Since we just calculated the former, and we can reuse the latter from our Column GPM
        ;; tests, the calculation of the constant is straightforward.
        ;; P(color = red, height = 6 | flip = true) = -3.27105108673 - (-0.4795730803) = -2.7914780064

        constrained-sol -2.7914780064
        constrained-logp (gpm.proto/logpdf view-inf targets-constrained constraints)

        ;; For the sake of floating point precision, we set an error threshold.
        threshold 1e-6]
    ;; Check isolated primitives.
    (is (utils/almost-equal? bernoulli-logp bernoulli-true-sol absolute-difference threshold))
    (is (utils/almost-equal? categorical-logp categorical-red-sol absolute-difference threshold))
    (is (utils/almost-equal? gaussian-logp gaussian-6-sol absolute-difference threshold))

    ;; Check unconstrained and constrained queries.
    (is (utils/almost-equal? unconstrained-logp unconstrained-sol absolute-difference threshold))
    (is (utils/almost-equal? constrained-logp constrained-sol absolute-difference threshold))))

(deftest simulate
  (let [targets-unconstrained ["color" "height" "flip"]
        empty-constraints {}
        targets-constrained ["color" "height"]
        constraints {"flip" true}
        ;; We verify that calls to simulate for a single column variable return the same results
        ;; as defined in the column tests for those primitive types and values. Different number of
        ;; samples are necessary to achieve the same level of accuracy.
        ;; Refer to `inferenceql.inference.gpm.column-test` for the mathematical explanation of the below.
        threshold-bernoulli 1e-2
        unconstrained-bernoulli-mean 0.619047619

        threshold-categorical 5e-2
        unconstrained-categorical-dist {"red" 0.4619047619 "blue" 0.269047619 "green" 0.269047619}

        threshold-gaussian 0.5
        unconstrained-gaussian-mean 3.2761904762

        ;; First, we consider an unconstrained simulation query. Because a category
        ;; is sampled before independently simulating each variable, we can treat n-samples
        ;; from a view as n-samples of each of the column variables, when considered independently.
        ;; That is, we can consider them as joint samples across all columns, or samples from each
        ;; column independently.
        n-samples 10000
        unconstrained-samples (repeatedly n-samples #(gpm.proto/simulate view-inf targets-unconstrained empty-constraints))
        unconstrained-bernoulli-emp-mean (double (utils/average (map (fn [sample] (if sample 1 0))
                                                                     (flatten (map #(get % "flip")
                                                                                   unconstrained-samples)))))
        unconstrained-categorical-emp-dist (reduce-kv (fn [m k v]
                                                        (assoc m k (double (/ v n-samples))))
                                                      {}
                                                      (frequencies (flatten (map #(get % "color")
                                                                                 unconstrained-samples))))
        unconstrained-gaussian-emp-mean (double (utils/average (flatten (map #(get % "height")
                                                                             unconstrained-samples))))
        ;; Next, we consider a constrained simulation query with two target variables, and
        ;; one constraint. Instead of weighting categories by just the CRP weights, we re-weight by
        ;; the logpdf of the constrained variable(s).

        constrained-samples (repeatedly n-samples #(gpm.proto/simulate view-inf targets-constrained constraints))
        ;; So, instead of normalized weights crp-0 crp-1 ... crp-n crp-alpha, we calculate the
        ;; unnormalized weights as follows:
        ;;    weights = 1/Z * (crp-0 * P(constraints | c_0) ... crp-alpha * P(constraints | c_alpha))
        ;; where Z is the normalizing constant, and c_i represents the ith category.
        ;; Therefore, we recalculate the means from `inferenceql.inference.gpm.column-test` to
        ;; account for this re-weighting.

        ;; Recall that the CRP weights are defined above as {:one 5/7, :two 1/7, :aux 1/7}.

        ;; weight-1-unnorm = 4/7 * P(true | c_1) = 4/7 * 5/6 = 0.4761904762
        ;; weight-2-unnorm = 2/7 * P(true | c_2) = 1/7 * 1/4 = 0.03571428571
        ;; weight-aux-unnorm = 1/7 * P(true | c_aux) = 1/7 * 1/2 = 0.07142857143
        ;; Z = weight-1-unnorm + weight-2-unnorm + weight-aux-unnorm
        ;;   = 0.4761904762 + 0.03571428571 + 0.07142857143
        ;;   = 0.5833333333

        ;; weight-1 = 0.4761904762 / 0.5833333333 = 0.8163265307
        ;; weight-2 = 0.03571428571 / 0.5833333333 = 0.06122448979
        ;; weight-aux = 0.07142857143 / 0.5833333333 = 0.1224489796

        ;; Note that this has made the first category much more likely than before!
        ;; Now we recalculate the necessary means.

        ;; "height" :gaussian
        ;; mu-1 = (r * m + sum-x-1) / (r + n) = (1 * 0 + 22)/ (1 + 4) = 4.4
        ;; mu-2 = (r * m + sum-x-2) / (r + n) = (1 * 0 + 8)/ (1 + 2) = 2.6666666667
        ;; mu-aux = (r * m) / r = 0
        ;; mean-mu = 0.8163265307 * 4.4 + 0.06122448979 * 2.6666666667 + 0.1224489796 * 0 = 3.7551020412
        constrained-gaussian-mean 3.7551020412
        constrained-gaussian-emp-mean (double (utils/average (flatten (map #(get % "height") constrained-samples))))

        ;; "color" :categorical
        ;; alpha-red-mean = 0.8163265307 * 6/10 + 0.06122448979 * 2/8 + 0.1224489796 * 1/3 = 0.5459183674
        ;; alpha-blue-mean = 0.8163265307 * 2/10 + 0.06122448979 * 3/8 + 0.1224489796 * 1/3 = 0.2270408163
        ;; alpha-green-mean = 0.8163265307 * 2/10 + 0.06122448979 * 3/8 + 0.1224489796 * 1/3 = 0.2270408163
        constrained-categorical-dist {"red" 0.5459183674 "blue" 0.2270408163 "green" 0.2270408163}
        constrained-categorical-emp-dist (reduce-kv (fn [m k v]
                                                      (assoc m k (double (/ v n-samples))))
                                                    {}
                                                    (frequencies (flatten (map #(get % "color")
                                                                               constrained-samples))))]
    (is (utils/almost-equal? unconstrained-bernoulli-emp-mean unconstrained-bernoulli-mean absolute-difference threshold-bernoulli))
    (is (utils/almost-equal-maps? unconstrained-categorical-emp-dist unconstrained-categorical-dist absolute-difference threshold-categorical))
    (is (utils/almost-equal? unconstrained-gaussian-emp-mean unconstrained-gaussian-mean absolute-difference threshold-gaussian))
    (is (utils/almost-equal-maps? constrained-categorical-emp-dist constrained-categorical-dist absolute-difference threshold-categorical))
    (is (utils/almost-equal? constrained-gaussian-emp-mean constrained-gaussian-mean absolute-difference threshold-gaussian))))

(def data-bernoulli
  [false false false false false false])

(def hypers-bernoulli
  {:alpha 1 :beta 1})

(def column-bernoulli
  (column/construct-column-from-latents "new-col"
                                        :bernoulli
                                        hypers-bernoulli
                                        latents
                                        (into {} (map-indexed vector data-bernoulli))
                                        {:crosscat true}))

;; Tests incorporate-column by adding and then removing the column, ensuring that
;; the initial and final view states are the same.
(deftest incorporate-column
  (let  [view' (view/incorporate-column view-inf column-bernoulli)
         view (view/unincorporate-column view' (:var-name column-bernoulli))]
    (is (= view  view-inf))))
