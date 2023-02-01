(ns inferenceql.inference.kernels.view
  (:require [inferenceql.inference.utils :as utils]
            [inferenceql.inference.gpm.utils :as gpm.utils]
            [inferenceql.inference.primitives :as primitives]
            [inferenceql.inference.gpm.crosscat :as xcat]))

(defn infer-column-view-xcat
  "Given a XCat GPM, returns the GPM with updated latent column-view assignments."
  [xcat m]
  (let [col-names (shuffle (-> xcat :latents :z keys))]
    ;; For each column in the model:
    ;; 1.) Remove the column from the current view.
    ;; 2.) Calculate the logpdf-score of that column under the latent
    ;;     assignments associated with that view.
    ;; 3.) Sample new view assignment for column, and
    ;;     update latents accordingly.
    (reduce (fn [xcat' col-name]
              (let [latents (:latents xcat')
                    z (get-in latents [:z col-name])
                    column (-> xcat' :views (get z) :columns (get col-name))
                    singleton? (= 1 (get-in latents [:counts z]))
                    m (if singleton? (dec m) m)
                    ;; Remove the current column from the model.
                    xcat-minus (-> xcat'
                                   (xcat/unincorporate-column col-name)
                                   ;; Add auxiliary views, if necessary. If z
                                   ;; represents a singleton category and m is 1,
                                   ;; then no new views are added and z is treated
                                   ;; like the auxiliary view (since it is now empty).
                                   (xcat/add-aux-views m))
                    ;; Get logpdf-score of each view if the current column were part of it.
                    lls (xcat/view-logpdf-scores xcat-minus column)
                    crp-weights (gpm.utils/crp-weights xcat-minus m)
                    logps {:p (utils/log-normalize (merge-with + lls crp-weights))}
                    z' (primitives/simulate :log-categorical logps)]
                (if (= z z')
                  xcat'
                  (-> xcat-minus
                      (xcat/incorporate-column column z')
                      (xcat/filter-empty-views)))))
            xcat
            col-names)))


(defn infer-single-column-view-xcat
  "Given a XCat GPM, returns the GPM with updated latent column-view assignments.
  THIS IS USED FOR PROBABILISTIC SEARCH
  "
  [xcat m col-name]
  (let [col-names [col-name]]
    ;; For each column in the model:
    ;; 1.) Remove the column from the current view.
    ;; 2.) Calculate the logpdf-score of that column under the latent
    ;;     assignments associated with that view.
    ;; 3.) Sample new view assignment for column, and
    ;;     update latents accordingly.
    (reduce (fn [xcat' col-name]
              (let [latents (:latents xcat')
                    z (get-in latents [:z col-name])
                    column (-> xcat' :views (get z) :columns (get col-name))
                    singleton? (= 1 (get-in latents [:counts z]))
                    m (if singleton? (dec m) m)
                    ;; Remove the current column from the model.
                    xcat-minus (-> xcat'
                                   (xcat/unincorporate-column col-name)
                                   ;; Add auxiliary views, if necessary. If z
                                   ;; represents a singleton category and m is 1,
                                   ;; then no new views are added and z is treated
                                   ;; like the auxiliary view (since it is now empty).
                                   (xcat/add-aux-views m))
                    ;; Get logpdf-score of each view if the current column were part of it.
                    lls (xcat/view-logpdf-scores xcat-minus column)
                    crp-weights (gpm.utils/crp-weights xcat-minus m)
                    ;logps {:p (utils/log-normalize (merge-with + lls crp-weights))}
                    ;z' (primitives/simulate :log-categorical logps)
                    logps (utils/log-normalize (merge-with + lls crp-weights))
                    ; XXX: this is currently picking the MAP. Unless One works
                    ; with ensembles, doing a Gibbs step is too unstable.
                    z' (key (apply max-key val logps))
                    _ (prn z')
                    ]
                (if (= z z')
                  xcat'
                  (-> xcat-minus
                      (xcat/incorporate-column column z')
                      (xcat/filter-empty-views)))))
            xcat
            col-names)))


(defn infer
  "Conducts column-view inference on a GPM.
  Supports XCat GPMs only."
  ([gpm]
   (infer gpm {:m 1}))
  ([gpm {:keys [m]}]
   (cond
     (xcat/xcat? gpm) (infer-column-view-xcat gpm m)
     :else (throw (ex-info (str "Column-view inference cannot operate"
                                " on GPM of type: "
                                (type gpm))
                           {:gpm-type (type gpm)})))))
