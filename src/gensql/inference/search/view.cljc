(ns gensql.inference.search.view
  (:require [gensql.inference.kernels.hyperparameters :as col-hypers]
            [gensql.inference.gpm.view :as view]
            [gensql.inference.gpm.column :as column] [gensql.inference.gpm.proto :as gpm.proto]))

(defn generate-logpdf
  "Given a view, variable name, and row-id, calculates the logpdf
  of the row evaluated at the specified variable name, constrained
  on other values contained in the row."
  [view var-name row-id]
  (let [columns (:columns view)
        row-data (apply merge (map (fn [[col-name col]]
                                     {col-name (get-in col [:data row-id])})
                                   columns))
        target {var-name true} ; Assumes binary label.
        constraints (dissoc row-data var-name)]
    (if (some? (get row-data var-name))
      {row-id (if (get row-data var-name) 0 ##-Inf)}
      {row-id (gpm.proto/logpdf view target constraints)})))

(defn generate-logpdfs
  "Given a view and variable name, returns for each row the logpdf
  of the specified variable name, conditioned on the rest of the
  values in that row."
  [var-name view]
  (let [row-ids (-> view :latents :y keys)]
    (map #(generate-logpdf view var-name %) row-ids)))

(defn search
  "Runs search on a DPMM with the CrossCat kernels.
   Assumes `view` is a View GPM, and `binary-labels` is a map of {row-id label}."
  [view binary-labels]
  (let [latents (:latents view)
        binary-name :new-col
        binary-column (column/construct-column-from-latents
                       binary-name
                       :bernoulli
                       {:alpha 0.5 :beta 0.5}
                       latents
                       binary-labels
                       {:crosscat true})]
    (->> binary-column
         (view/incorporate-column view)
         (col-hypers/infer-column-view binary-name)
         (generate-logpdfs binary-name)
         (apply merge)
         (sort-by second >))))
