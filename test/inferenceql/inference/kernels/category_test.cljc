(ns inferenceql.inference.kernels.category-test
  (:require [clojure.test :as test :refer [deftest is]]
            [inferenceql.inference.gpm.view :as view]
            [inferenceql.inference.kernels.category :as c]))

(def latents-one-mislabeled
  {:alpha 1
   :counts {0 4 1 6}
   :y {0 1
       1 0
       2 0
       3 0
       4 0
       5 1
       6 1
       7 1
       8 1
       9 1}})

(def data
  {0 {"color" "red" "height" 10}
   1 {"color" "red" "height" 10}
   2 {"color" "red" "height" 10}
   3 {"color" "red" "height" 10}
   4 {"color" "red" "height" 10}
   5 {"color" "blue" "height" -10}
   6 {"color" "blue" "height" -10}
   7 {"color" "blue" "height" -10}
   8 {"color" "blue" "height" -10}
   9 {"color" "blue" "height" -10}})

(def view {:hypers {"color"  {:alpha 1}
                    "height" {:m 0 :r 1 :s 1 :nu 1}}})

(def types {"color"  :categorical
            "height" :gaussian})

(def options {"color" ["red" "blue" "green"]})

;; Tests `infer-row-category-view` by specifying two distinct categories with the target row
;; labeled in the wrong category. The output should contains a `latents-l` structure
;; that reflects the target row being moved to the other category at roughly 95% of the time.
(deftest infer-row-category-view-mislabeled
  (let [dpmm (view/construct-view-from-latents view latents-one-mislabeled types data {:options options :crosscat true})
        ;; The row of interest has row-id 0 and data {"color" "red", "height" 10}.

        ;; logPs for categories c_0, c_1, c_aux for datum = {"color" "red", "height" 10}.
        ;; logP["color" = "red", "height" = 10 | c_0]
        ;;    = log(weight_0 * P["color" = "red" | c_0]) + log(weight_0 * P["height" = 10 | c_0])
        ;;    = log(3 / 10) + -0.2006706954621511 + -2.573270124731785
        ;;   ~= (math/exp -3.9779136245198723)
        ;;     -> exp: 0.018724665294720857

        ;; logP["color" = "red", "height" = 10 | c_1]
        ;;    = log(weight_1 * P["color" = "red" | c_1]) + log(weight_1 * P["height" = 10 | c_1])
        ;;    = log(6 / 10) + -1.6094379124341003 + -4.838494052083643
        ;;   ~= -6.958757588283734
        ;;    -> exp: 9.502764760549094E-4

        ;; logP["color" = "red", "height" = 10 | c_aux]
        ;;    = log(weight_aux * P["color" = "red" | c_aux]) + log(weight_1 * P["height" = 10 | c_aux])
        ;;    = log(1 / 10) + -1.0986122886681096 + -5.423129108853699
        ;;   ~= -8.824326490515855
        ;;     -> exp: 1.471105091759976E-4

        ;; Normalizing these probabilities yields the following distribution over categories:
        ;; {c_0 0.9446380743158054
        ;;  c_1 0.047940367759802
        ;;  c_aux 0.007421557924392457}

        ;; This implies that the transition should occur roughly 95% of the time.
        ;; We adjust the switch-% below to be 93% to give some leeway for our sample size.

        ;; We also test with 2 auxiliary categories, which results in identical transition probabilities,
        ;; due to the nature of the auxiliary weightings.

        m1 1 ; One auxiliary category.
        m2 2 ; Two auxiliary categories.
        row-id 0 ; Target rowid.
        desired-y' 0 ; The mislabeled point should go from category 1 to 0.
        switch-% 0.93 ; The threshold proporation of switching to the other category.
        iters 500

        y-m1' (frequencies (repeatedly iters #(-> dpmm (c/infer {:m m1}) :latents :y (get row-id))))
        y-m2' (frequencies (repeatedly iters #(-> dpmm (c/infer {:m m2}) :latents :y (get row-id))))]

   ;; m = 1
   (is (>= (get y-m1' desired-y') (* switch-% iters)))

   ;; m = 2
   (is (>= (get y-m2' desired-y') (* switch-% iters)))))
