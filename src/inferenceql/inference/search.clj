(ns inferenceql.inference.search
  (:require [inferenceql.inference.search.view :as search.view]
            [inferenceql.inference.gpm.view :as view]))

(def latents
  {:alpha 0.1
   :counts {0 6 1 4}
   :y {0 0
       1 0
       2 0
       3 0
       4 0
       5 0
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
   5 {"color" "red" "height" 6}
   6 {"color" "green" "height" 3}
   7 {"color" "blue" "height" 3}
   8 {"color" "blue" "height" 3}
   9 {"color" "green" "height" 3}})

(def view {:hypers {"color"  {:alpha 2}
                    "height" {:m 0 :r 1 :s 1 :nu 1}}})

(def types {"color"  :categorical
            "height" :gaussian})

(def options {"color" ["red" "blue" "green"]})

(def dpmm (view/construct-view-from-latents view latents types data {:options options :crosscat true}))

(def binary-data-1-correct-each
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

;; Expect high probability for one category, and low probability for other.
(search.view/search dpmm binary-data-1-correct-each)

(def binary-data-1-correct-each-1-incorrect
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

;; Expect low-ish probability for one category, and lower probability for other.
(search.view/search dpmm binary-data-1-correct-each-1-incorrect)

(def binary-data-1-correct-each-1-incorrect-each
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

;; Expect perfectly uncertain probability from both.
(search.view/search dpmm binary-data-1-correct-each-1-incorrect-each)
