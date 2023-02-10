(ns inferenceql.inference.search.crosscat
  #_(:import [java.io PushbackReader])
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [inferenceql.inference.kernels.hyperparameters :as col-hypers]
            [inferenceql.inference.kernels.view :as view-kernel]
            [inferenceql.inference.gpm.view :as view]
            [clojure.math :as math]
            [inferenceql.inference.gpm.utils :as gpm.utils]
            [inferenceql.inference.gpm.column :as column]
            [inferenceql.inference.gpm :as gpm]
            [inferenceql.inference.gpm.crosscat :as crosscat]
            [inferenceql.inference.gpm.multimixture.metrics :as metrics]
            [inferenceql.inference.utils :as utils]))

(defn get-largest-view
  "Given an `xcat` gpm, this returns the view-id and view with the largest number of columns.
  This corresponds to a MAP estimate of the current posterior by the CRP.
  The view returned is consistent when there is a tie."
  [xcat]
  (let [;; Get the views in a consistent order.
        views (sort-by key (:views xcat))
        column-counts (map #(count (:columns (val %)))
                           (sort-by key (:views xcat)))]
    (->> (map vector views column-counts)
         (sort-by second >)
         (first) ; first pair
         (first)))) ; the view MapEntry.

(defn incorporate-labels
  "Returns a XCat gpm that has a new :label column GPM incorporated based on `binary-labels`.
  Args:
    xcat: An XCat gpm
    binary-labels: A map of row-id to bool. Not all row-ids need to be labeled.
  The returned GPM will be able to perform few-shot learning search based on `binary-labels`"
  [xcat binary-labels]
  (let [;; NOTE: Currently, the new :label column is always added to the view
        ;; with the most columns in the XCat gpm.
        [view-id view] (get-largest-view xcat)
        latents (:latents view)
        binary-name :label
        param 0.1
        binary-column (column/construct-column-from-latents
                       binary-name
                       :bernoulli
                       {:alpha param :beta param}
                       latents
                       binary-labels
                       {:crosscat true})]
    (view-kernel/infer-single-column-view-xcat (as-> binary-column $
          ;; Incorporate column.
          (crosscat/incorporate-column xcat $ view-id)
          ;; Perform inference.
          ;;(col-hypers/infer-column-xcat binary-name $)
          ) 1 :label)))

(defn columns-in-view
  [view model]
  (keys (:columns (view (:views model)))))

(defn column-view-map
  [model]
  (->> (keys (:views model))
       (map (fn [v] (map (fn [col] [col v]) (columns-in-view v model))))
       (apply concat)
       (into {})))

(defn cluster-probabilities [view data]
  (let [m 1 ;; XXX
        crp-weights (gpm.utils/crp-weights view m)
        row-lls (map #(-> view (:columns) (view/column-logpdfs %)) data) ]
    (if (= row-lls `({}))
      crp-weights
      (let [lls (apply merge-with + row-lls)
            _ (println "lls")
            _ (prn lls)
            ]
        (utils/log-normalize (merge-with + lls crp-weights))))))


(defn p-vectors [logp-map1 logp-map2]
  (let [ids (keys logp-map1)
        p (mapv #(math/exp (get logp-map1 %)) ids)
        q (mapv #(math/exp (get logp-map2 %)) ids)]
      [p q]))

(defn relevance-distance
  [gpm current-row comparison-rows view-indicator-col]
  (let [column-view-assignments (column-view-map gpm)
        ; XXX: need to unincorporate the current row, if comes from data!!!
        view-k (get column-view-assignments view-indicator-col)
        logp-map1 (cluster-probabilities (view-k (:views gpm)) [current-row])
        logp-map2 (cluster-probabilities (view-k (:views gpm)) comparison-rows)
        [p q] (p-vectors logp-map1 logp-map2)]
  (metrics/jensen-shannon-divergence p q)))

(defn p-same-cluster [logp-map1 logp-map2]
  (let [ids (keys logp-map1)]
    (->>
      (mapv
        (fn [k1]
          (mapv (fn [k2] (when (= k1 k2) (+ (get logp-map1 k1)  (get logp-map2 k2))))
                ids))
        ids)
      (apply concat)
      (remove nil?)
      (map math/exp)
      (apply + ))))


(defn relevance-probability
  [gpm current-row comparison-rows view-indicator-col]
  (let [
        column-view-assignments (column-view-map gpm)
        ; XXX: need to unincorporate the current row, if comes from data!!!
        view-k (get column-view-assignments view-indicator-col)
        _ (println "view-k")
        _ (prn view-k)
        logp-map1 (cluster-probabilities (view-k (:views gpm)) [current-row])
        logp-map2 (cluster-probabilities (view-k (:views gpm)) comparison-rows)
        _ (println "log probabilities")
        _ (println "current row")
        _ (prn logp-map1)
        _ (println "comparison")
        _ (prn logp-map2)
        _ (println "------")]
    (p-same-cluster logp-map1 logp-map2)))


#_(def model (edn/read {:readers gpm/readers} (PushbackReader. (io/reader "sample.test.edn"))))

;; not similar
#_(relevance-probability model
                       {:x 10} ;; current row
                       [{:y 107} {:x 4.7}] ;; rows to compare with
                       :x ;; context column -- indicating view.
                       )

;; similar
#_(relevance-probability model
                       {:x 0} ;; current row
                       [{:y 107} {:x 4.7}] ;; rows to compare with
                       :x ;; context column -- indicating view.
                       )

;; similar -- less relevance prob
#_(relevance-probability model
                       {:x 0} ;; current row
                       [{:y 107} ] ;; rows to compare with
                       :x ;; context column -- indicating view.
                       )

;; similar -- more relevance prob
#_(relevance-probability model
                       {:x 0} ;; current row
                       [{:y 107} {:x 4.7}
                        {:y 107} {:x 4.7}
                       ] ;; rows to compare with
                       :x ;; context column -- indicating view.
                       )

;; The next two are counter-intuitive but expeced.
;; this has a higher relevance probability.
#_(relevance-probability model
                       {:x 0} ;; current row
                       [{:x 4.7} ] ;; rows to compare with
                       :x ;; context column -- indicating view.
                       )
;; than this... because it's not as clear which generative process the current
;; row stems from.
#_(relevance-probability model
                       {:x 4.7} ;; current row
                       [{:x 4.7} ] ;; rows to compare with
                       :x ;; context column -- indicating view.
                       )

;;;; searching via distance

#_(relevance-distance model
                       {:x 0}
                       [{:y 107} {:x 4.7}]
                       :x) ;; this is fine

#_(relevance-distance model
                       {:x 0}
                       [{:y 107} {:x 4.7}]
                       :a) ;; this is not -- because the both rows have 0 info on the relevant view, both return the prior. Distance is zero.
