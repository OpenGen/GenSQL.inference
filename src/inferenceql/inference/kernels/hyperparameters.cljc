(ns inferenceql.inference.kernels.hyperparameters
  (:require [inferenceql.inference.gpm.column :as column]
            [inferenceql.inference.gpm.view :as view]
            [inferenceql.inference.utils :as utils]
            [inferenceql.inference.primitives :as primitives]))

(defn sample-hyper-parameter
  "Given a Column GPM and a hyperparameter name, samples a new value of the
  associated hyperparameter from the posterior approximated by the Column's
  hyper-grid attribute."
  [column hyper-name]
  (let [hyper-grid (get-in column [:hyper-grid hyper-name])
        ;; For each value in the hyper-grid, calculate the logpdf-score
        ;; of the column across all categories with the updated value.
        scores (reduce (fn [scores' hyper-value]
                         (assoc scores'
                                hyper-value
                                (column/crosscat-logpdf-score
                                  (assoc-in column [:hyperparameters hyper-name] hyper-value))))
                       {}
                       hyper-grid)
        normalized-scores (utils/log-normalize scores)
        logps {:p normalized-scores}]
    (primitives/simulate :log-categorical logps)))

(defn infer-hyperparameters-column
  "Column hyperparameter inference.
  Sample a value for each hyperparameter in order using Gibbs sampling
  on a discretized (approximated) posterior space, and update the corresponding
  value in the Column GPM."
  [column]
  (let [inferred-column (reduce (fn [column' hyper-name]
                                  (let [hyper' (sample-hyper-parameter column' hyper-name)]
                                    (assoc-in column' [:hyperparameters hyper-name] hyper')))
                                column
                                (keys (:hyperparameters column)))]
    ;; Propagate the updated hyperparameters from the Column level
    ;; to each of the constituent pGPMs. This is done to ensure consistency
    ;; across all GPMs after inference.
    (column/update-hypers inferred-column)))

(defn infer-hyperparameters-view
  "Conducts column hyperparameter inference on each of the Column GPMs
  contained in the given View GPM.
  This inference can be performed in parallel across all columns."
  [view]
  (reduce (fn [view' col-name]
            (update-in view'
                       [:columns col-name]
                       #(infer-hyperparameters-column %)))
          view
          (keys (:columns view))))

(defn infer-hyperparameters-xcat
  "Conducts column hyperparameter inference on each of the Column GPMs
  contained in each of the View GPMs in the given XCat GPM."
  [xcat]
  (reduce (fn [xcat' view-name]
            (update-in xcat'
                       [:views view-name]
                       #(infer-hyperparameters-view %)))
          xcat
          (keys (:views xcat))))

(defn infer-column
  "Conducts column hyperparameter inference on only the column of
  interest for a given View GPM."
  [column-name view]
  (update-in view [:columns column-name]
             #(infer-hyperparameters-column %)))

(defn infer
  "Conducts hyperparameter inference on a GPM.
  Supports Column GPMs only."
  [gpm]
  (cond
    (column/column? gpm) (infer-hyperparameters-column gpm)
    (view/view? gpm) (infer-hyperparameters-view gpm)
    ;; The below will be uncommented as the necessary GPMs are introduced.
    ; (xcat/->XCat) (infer-hyperparameters-xcat gpm)
    :else (throw (ex-info (str "Column hyperparameter inference cannot operate"
                               " on GPM of type: "
                               (type gpm))
                          {:gpm-type (type gpm)}))))
