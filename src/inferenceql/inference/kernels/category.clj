(ns inferenceql.inference.kernels.category
  (:require [inferenceql.inference.utils :as utils]
            [inferenceql.inference.primitives :as primitives]
            [inferenceql.inference.gpm.view :as view]))

(defn save-latents-raw
  [latents file]
  (spit file (str latents "\n") :append true))

(defn crp-weights
  "Given a view and the number of auxiliary categories, calculates the CRP weights."
  [view m]
  (let [latents (:latents view)
        alpha (:alpha latents)
        counts (:counts latents)
        z (apply + (vals counts))
        ;; The below check is added to avoid divide-by-zero errors,
        ;; in the event that no new categories are added, but an empty existing
        ;; category is used in its place.
        m (if (zero? m) (inc m) m)
        altered-counts (reduce-kv (fn [counts' category-name cnt]
                                    ;; Set the auxiliary weight from 0 to (alpha / m) / z
                                    ;; which for m auxiliary categories, will sum to alpha / z.
                                    (let [cnt' (if (zero? cnt) (/ alpha m) cnt)]
                                      (assoc counts' category-name (Math/log (/ cnt' z)))))
                                  {}
                                  counts)]
    (utils/log-normalize altered-counts)))

(defn infer-row-category-view
  "Given a view and list of row-ids, infers the assignment of each row with both the
  current categories in the view, as well as `m` specified auxiliary ones.
  This is Algorithm 8 from http://www.stat.columbia.edu/npbayes/papers/neal_sampling.pdf"
  [view m]
  (let [row-ids (shuffle (-> view :latents :y keys))]
    ;; For each row in the view:
    ;; 1.) Remove the row from the current category.
    ;; 2.) Calculate the logpdf of that row being generated
    ;;     by each category and weight with CRP weighting.
    ;; 3.) Sample new category assignment for row, and
    ;;     update latents accordingly.
    (reduce (fn [view' row-id]
              (let [row-data (view/get-data view' row-id)
                    latents (:latents view')
                    y (get-in latents [:y row-id])
                    singleton? (= 1 (get-in latents [:counts y]))
                    m (if singleton? (dec m) m)
                    ;; Remove the current row from the model.
                    view-minus (-> view'
                                   (view/unincorporate-from-category row-data y row-id)
                                   ;; Add auxiliary categories, if necessary. If y
                                   ;; represents a singleton category and m is 1,
                                   ;; then no new categories are added and y is treated
                                   ;; like the auxiliary category (since it is now empty).
                                   (view/add-aux-categories m))
                    ;; Get logpdf that each category generated the datum.
                    lls (-> view-minus
                            (:columns)
                            (view/column-logpdfs row-data))
                    crp-weights (crp-weights view-minus m)
                    logps {:p (utils/log-normalize (merge-with + lls crp-weights))}
                    y' (primitives/simulate :log-categorical logps)]
                (if (= y y')
                  view'
                  (-> view-minus
                      (view/incorporate-into-category row-data y' row-id)
                      (view/filter-empty-categories)))))
    view
    (seq row-ids))))

(defn infer-row-category-xcat
  "Given a CrossCat model, returns the model with updated latent row-category assignments."
  [xcat & {:keys [m] :or {m 1}}]
  (let [views (:views xcat)]
    (reduce-kv (fn [model view-name _]
                 (update-in model [:views view-name]
                            (fn [view]
                              (infer-row-category-view view m))))
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
     ;; The below will be uncommented as the necessary GPMs/implementations of inference are introduced.
     ; (column/column? gpm) (infer-hyperparameters-column gpm)
     ; (xcat/->XCat) (infer-row-category-xcat gpm)
     :else (throw (ex-info (str "Row category inference cannot operate"
                                " on GPM of type: "
                                (type gpm))
                           {:gpm-type (type gpm)})))))
