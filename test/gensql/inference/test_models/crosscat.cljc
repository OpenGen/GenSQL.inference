(ns gensql.inference.test-models.crosscat
  (:require [gensql.inference.gpm.crosscat :as xcat]))

;;; A two view XCat model with "color" and "height" in view-1 and "flip" in view-2.

(def data
  {0 {:color "red" :height 6 :flip true}
   1 {:color "red" :height 6 :flip true}
   2 {:color "red" :height 6 :flip true}
   3 {:color "red" :height 4 :flip false}
   4 {:color "blue" :height 4 :flip false}
   5 {:color "green" :height 4 :flip false}})

(def model
  (let [options {:color ["red" "blue" "green"]}
        view-1-name (gensym)
        view-2-name (gensym)

        xcat-spec {:views {view-1-name {:hypers {:color  {:alpha 2}
                                                 :height {:m 0 :r 1 :s 2 :nu 3}}}
                           view-2-name {:hypers {:flip  {:alpha 1 :beta 1}}}}
                   :types {:color  :categorical
                           :height :gaussian
                           :flip :bernoulli}}

        xcat-latents {:global {:alpha 0.5}
                      :local {view-1-name {:alpha 1
                                           :counts {:one 4 :two 2}
                                           :y {0 :one
                                               1 :one
                                               2 :one
                                               3 :one
                                               4 :two
                                               5 :two}}
                              view-2-name {:alpha 1
                                           :counts {:one 3 :two 3}
                                           :y {0 :one
                                               1 :one
                                               2 :one
                                               3 :two
                                               4 :two
                                               5 :two}}}}]
    (xcat/construct-xcat-from-latents xcat-spec xcat-latents data {:options options})))
