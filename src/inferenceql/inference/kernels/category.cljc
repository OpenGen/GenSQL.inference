(ns inferenceql.inference.kernels.category
  (:require [inferenceql.inference.gpm.view :as view]
            [inferenceql.inference.gpm.crosscat :as xcat]))

#?(:clj
   (defn save-latents-raw
     [latents file]
     (spit file (str latents "\n") :append true)))

(defn infer-row-category-view
  "Given a view and list of row-ids, infers the assignment of each row with both the
  current categories in the view, as well as `m` specified auxiliary ones."
  [view m]
  (view/infer-row-category-view view m))

(defn infer-row-category-xcat
  "Given a CrossCat model, returns the model with updated latent row-category assignments."
  [xcat m]
  (let [views (:views xcat)]
    (reduce-kv (fn [model view-name view]
                 (assoc-in model [:views view-name] (infer-row-category-view view m)))
               xcat
               views)))

(defn infer
  "Conducts row-category inference on a GPM.
  Supports View GPMs only."
  ([gpm]
   (infer gpm {:m 1}))
  ([gpm {:keys [m]}]
   (cond
     (view/view? gpm) (infer-row-category-view gpm m)
     (xcat/xcat? gpm) (infer-row-category-xcat gpm m)
     ;; The below will be uncommented as the necessary GPMs/implementations of inference are introduced.
     ; (column/column? gpm) (infer-hyperparameters-column gpm)
     :else (throw (ex-info (str "Row category inference cannot operate"
                                " on GPM of type: "
                                (type gpm))
                           {:gpm-type (type gpm)})))))
