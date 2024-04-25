(ns gensql.inference.gpm.insert-test
  (:require [gensql.inference.gpm.view :as view]
            [clojure.test :as test :refer [deftest is]]
            [gensql.inference.gpm.proto :as gpm.proto]))


;; Define a DPMM.
(def latents
  {:alpha 0.1
   :counts {:one 3 :two 1}
   :y {0 :one
       1 :one
       2 :one
       3 :two}})

(def data
  {0 {"x"  0.2}
   1 {"x"  0.0}
   2 {"x"  0.1}
   3 {"x"  10.}})

(def view-spec {:hypers {"x"  {:m 0 :r 1 :s 1 :nu 0.1}}})

(def types {"x"  :gaussian })

(def dpmm (view/construct-view-from-latents view-spec
                                            latents
                                            types
                                            data
                                            {:options nil :crosscat true}))
(deftest insert
  ;; Test whether we can insert a row into an existing DPMM correctly.
  (let [obs-rows [0 1 2 3]
        new-row {"x" 10.01}
        n-samples 1000
        ;; Use incorporate and the get frequencies from assigning tables from
        ;; the CRP predictive.
        freq_incorporate (get (frequencies (repeatedly
                                             n-samples
                                             #(first (vals (apply
                                                             dissoc
                                                             (:y (:latents (gpm.proto/incorporate dpmm new-row)))
                                                             obs-rows)))))
                              :one
                              0)
        freq_insert  (get (frequencies (repeatedly
                                         n-samples
                                         #(first (vals (apply
                                                         dissoc
                                                         (:y (:latents (gpm.proto/insert dpmm new-row)))
                                                         obs-rows)))))
                          :two
                          0)]
    ;; The CRP predictive is poised towards cluster "one".
    (is (>= freq_incorporate 700))
    ;; The DPMM posterior is poised towards cluster "two".
    (is (>= freq_insert 700))))
