(ns gensql.inference.kernels.concentration-hypers-test
  (:require [clojure.test :as test :refer [deftest is]]
            [gensql.inference.kernels.concentration-hypers :as ch]
            [gensql.inference.utils :as utils]
            [gensql.inference.gpm.view :as view]))

(def latents-one-category
  {:alpha 1
   :counts {0 10}
   :y {0 0
       1 0
       2 0
       3 0
       4 0
       5 0
       6 0
       7 0
       8 0
       9 0}})

(def latents-three-categories
  {:alpha 1
   :counts {0 3 1 3 2 4}
   :y {0 0
       1 0
       2 0
       3 1
       4 1
       5 1
       6 2
       7 2
       8 2
       9 2}})

(def latents-ten-categories
  {:alpha 1
   :counts (zipmap (range 10) (repeat 1))
   :y (zipmap (range) (range 10))})

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

(def data-singleton
  {0 {"color" "red" "height" 10}
   1 {"color" "red" "height" 10}
   2 {"color" "red" "height" 10}
   3 {"color" "red" "height" 10}
   4 {"color" "red" "height" 10}
   5 {"color" "red" "height" 10}
   6 {"color" "red" "height" 10}
   7 {"color" "red" "height" 10}
   8 {"color" "red" "height" 10}
   9 {"color" "red" "height" 10}})

(def latents-singleton-misplace
  {:alpha 1
   :counts {0 9 1 1}
   :y {0 1
       1 0
       2 0
       3 0
       4 0
       5 0
       6 0
       7 0
       8 0
       9 0}})

(def view-spec {:hypers {"color"  {:alpha 1}
                         "height" {:m 0 :r 1 :s 1 :nu 1}}})

(def types {"color"  :categorical
            "height" :gaussian})

(def options {"color" ["red" "blue" "green"]})

(deftest infer-alpha
  (let [view-low-alpha (view/construct-view-from-latents
                        view-spec
                        latents-one-category
                        types
                        data
                        {:options options})
        view-medium-alpha (view/construct-view-from-latents
                           view-spec
                           latents-three-categories
                           types
                           data
                           {:options options})
        view-high-alpha (view/construct-view-from-latents
                         view-spec
                         latents-ten-categories
                         types
                         data
                         {:options options})
        n-iters 1000
        low-alpha (utils/average (map #(-> % :latents :alpha)
                                      (repeatedly n-iters #(ch/infer view-low-alpha))))
        medium-alpha (utils/average (map #(-> % :latents :alpha)
                                         (repeatedly n-iters #(ch/infer view-medium-alpha))))
        high-alpha (utils/average (map #(-> % :latents :alpha)
                                       (repeatedly n-iters #(ch/infer view-high-alpha))))]
    ;; The below are rough estimates for identifying low, medium, and high values of alpha.
    (is (< low-alpha 0.5))
    (is (<  1 medium-alpha 1.5))
    (is (<  7 high-alpha))))
