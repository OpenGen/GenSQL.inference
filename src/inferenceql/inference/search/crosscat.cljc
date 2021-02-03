(ns inferenceql.inference.search.crosscat
  (:require [inferenceql.inference.kernels.hyperparameters :as col-hypers]
            [inferenceql.inference.gpm.column :as column]
            [inferenceql.inference.gpm.crosscat :as crosscat]))

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
        binary-column (column/construct-column-from-latents
                       binary-name
                       :bernoulli
                       {:alpha 0.5 :beta 0.5}
                       latents
                       binary-labels
                       {:crosscat true})]
    (as-> binary-column $
          ;; Incorporate column.
          (crosscat/incorporate-column xcat $ view-id)
          ;; Perform inference.
          (col-hypers/infer-column-xcat binary-name $))))
