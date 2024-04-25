(ns gensql.inference.search.view-test
  (:require [clojure.test :refer [deftest is]]
            [gensql.inference.search.view :as search.view]
            [gensql.inference.utils :as utils]
            [gensql.inference.gpm.view :as view]))

(def latents
  {:alpha 0.1
   :counts {0 5 1 5}
   :y {0 0
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
  {0 {"color" "red" "height" 6}
   1 {"color" "red" "height" 6}
   2 {"color" "red" "height" 6}
   3 {"color" "red" "height" 6}
   4 {"color" "red" "height" 6}
   5 {"color" "green" "height" 3}
   6 {"color" "green" "height" 3}
   7 {"color" "green" "height" 3}
   8 {"color" "green" "height" 3}
   9 {"color" "green" "height" 3}})

(def view {:hypers {"color"  {:alpha 2}
                    "height" {:m 0 :r 1 :s 1 :nu 1}}})

(def types {"color"  :categorical
            "height" :gaussian})

(def options {"color" ["red" "blue" "green"]})

(def dpmm (view/construct-view-from-latents view latents types data {:options options :crosscat true}))

(def binary-data-1-correct-label-each
  {0 true    ; <= correct true label in one category
   1 nil
   2 nil
   3 nil
   4 nil
   5 nil
   6 nil
   7 nil
   8 nil
   9 false}) ; <= correct false label in one category

;; We  expect the search probability of the category containing a true label
;; to be greater than the other category, which contains only a false label.
(deftest search-dpmm-1-correct-label-each
  (let [high-prob-rows (set (range 1 5))
        low-prob-rows (set (range 5 9))
        search-results (into {} (search.view/search dpmm binary-data-1-correct-label-each))
        high-probs (into {} (select-keys search-results high-prob-rows))
        low-probs (into {} (select-keys search-results low-prob-rows))
        high-prob (apply min (vals high-probs))
        low-prob (apply max (vals low-probs))]
    ;; Verifies every element in one category has a higher search probability
    ;; than the other category.
    (is (> high-prob low-prob))))

(def binary-data-1-correct-label-each-1-incorrect
  {0 true    ; <= correct true label in one category
   1 false   ; <= incorrect false label in one category
   2 nil
   3 nil
   4 nil
   5 nil
   6 nil
   7 nil
   8 nil
   9 false}) ; <= correct false label in one category

;; We expect low-ish probability for one category which contains an equal number of
;; true and false labels, and lower probability for the other, which contains only a false label.
(deftest search-dpmm-1-correct-label-each-1-incorrect-label
  (let [high-prob-rows (set (range 2 5))
        low-prob-rows (set (range 5 9))
        search-results (into {} (search.view/search dpmm binary-data-1-correct-label-each-1-incorrect))
        high-probs (into {} (select-keys search-results high-prob-rows))
        low-probs (into {} (select-keys search-results low-prob-rows))
        high-prob (apply min (vals high-probs))
        low-prob (apply max (vals low-probs))]
    ;; Verifies every element in one category has a higher search probability
    ;; than the other category.
    (is (> high-prob low-prob))))

(def binary-data-1-correct-label-each-1-incorrect-label-each
  {0 true    ; <= correct true label
   1 false   ; <= incorrect false label
   2 nil
   3 nil
   4 nil
   5 nil
   6 nil
   7 nil
   8 true    ; <= incorrect true label
   9 false}) ; <= correct false label

;; We expect all probabilities to be the same (not necessarily perfect uncertainty, due
;; to column hyperparameter inference), since each category contains an equal number
;; of true and false observations. This currently isn't the case, as unequal values of
;; alpha and beta for the new bernoulli column lead to floating point imprecision and
;; the cause is unknown. https://github.com/OpenGen/GenSQL.inference/issues/46
(deftest search-dpmm-1-correct-label-each-1-incorrect-label-each
  (let [search-results (into {} (filter #(not (or (zero? (second %))
                                                  (= ##-Inf (second %))))
                                              (search.view/search dpmm binary-data-1-correct-label-each-1-incorrect-label-each)))
        ;; The below will be modified once the above issue is resolved.
        min-prob (apply min (map second search-results))
        max-prob (apply max (map second search-results))
        threshold 1e-2]
    (is (utils/almost-equal? min-prob max-prob utils/relerr threshold))))
