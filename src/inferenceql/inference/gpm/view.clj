(ns inferenceql.inference.gpm.view
  "Implementation of a GPM that represents a population of data of potentially
  different primitive types, but the same row categorizations. For a tabular dataset,
  this GPM abstracts a subset of columns of that dataset, the rows of which are clustered
  together in categories. This can also be considered as Dirichlet Process Mixture Model (DPMM).
  See `inferenceql.inference.gpm/view` for details."
  (:require [inferenceql.inference.utils :as utils]
            [inferenceql.inference.primitives :as primitives]
            [inferenceql.inference.gpm.column :as column]
            [inferenceql.inference.gpm.proto :as gpm.proto]))

(defn column-logpdfs
  "Given a map of columns, and targets, returns a map of category probabilities of the targets."
  [columns targets]
  (->> targets
       (map (fn [[var-name target]]
              (column/category-logpdfs (get columns var-name) {var-name target})))
       (apply merge-with + {})))

(defn add-aux-categories
  "Add m auxiliary categories to the given view."
  [view m]
  (reduce (fn [view' _]
            (let [symb (gensym)]
              (-> view'
                  ;; Make sure counts are zero for the new categories.
                  (assoc-in [:latents :counts symb] 0)
                  (update :columns (fn [col-dict]
                                     (reduce (fn [col-dict' [var-name col]]
                                               (assoc col-dict' var-name (column/add-category col symb)))
                                             {}
                                             col-dict))))))
          view
          (range m)))

(defn get-data
  "Grabs data from view by row-id."
  [view row-id]
  (reduce-kv (fn [x var-name column]
               (assoc x var-name (get-in column [:data row-id])))
             {}
             (:columns view)))

(defn filter-empty-categories
  "Filters empty categories from the given view."
  [view]
  (let [to-remove (map first (filter #(zero? (second %)) (get-in view [:latents :counts])))]
    (reduce (fn [view' category-to-remove]
              (-> view'
                  (update-in [:latents :counts] dissoc category-to-remove)
                  (update :columns (fn [col-dict]
                                     (reduce (fn [col-dict' var-name]
                                               (update-in col-dict' [var-name :categories]
                                                                    dissoc
                                                                    category-to-remove))
                                             col-dict
                                             (keys col-dict)))))) ;; XXX: This feels gross but it works.
            view
            to-remove)))

(defn incorporate-into-category
  "Incorporate method for CrossCat inference machinery.
  Incorporates `values` into the category specified by `category-key`."
  [view values category-key]
  (let [row-id (gensym)]
    (-> view
        ;; latents are used only in CrossCat inference. This implementation of row assignments
        ;; may be deprecated, depending on how the the inference procedures are adjusted.
        (assoc-in [:latents :y row-id] category-key)
        (update-in [:latents :counts category-key] (fnil inc 0))
        ;; assignments is used for a standalone View.
        (update-in [:assignments values] (fnil (fn [categories]
                                                 (-> categories
                                                     (update category-key (fnil inc 0))
                                                     (update :row-ids (fnil #(conj % row-id) #{}))))
                                               {}))
        (update :columns #(reduce-kv (fn [columns' col-name column]
                                       (let [col-data {col-name (get values col-name)}]
                                         (assoc columns' col-name (column/crosscat-incorporate
                                                                    column
                                                                    col-data
                                                                    category-key
                                                                    row-id))))
                                     {}
                                     %)))))

(defn unincorporate-from-category
  "Unincorporate method for CrossCat inference machinery.
  Unincorporates `values` associated with from the category
  specified by `category-remove`."
  ([view values category-remove]
   (let [row-id-remove (rand-nth (seq (get-in view [:assignments values :row-ids])))]
     (unincorporate-from-category view values category-remove row-id-remove)))
  ([view values category-remove row-id-remove]
   (-> view
       ;; Update latent counts and remove empty categories, if necessary. Again, this implementation
       ;; of row assignments may be deprecated, depending on how the inference procedures are adjusted.
       (update-in [:latents :y] dissoc row-id-remove)
       (update-in [:latents :counts category-remove] dec)
       (update :assignments (fn [values-dict]
                                (let [new-count (dec (get-in values-dict [values category-remove]))]
                                  (if (zero? new-count)
                                    (dissoc values-dict category-remove)
                                    (assoc values-dict category-remove new-count)))))
       (update-in [:assignments values :row-ids] #(disj % row-id-remove))
       (update :columns #(reduce-kv (fn [columns' col-name column]
                                      (assoc columns' col-name (column/crosscat-unincorporate
                                                                column
                                                                category-remove
                                                                row-id-remove)))
                                    {}
                                    %)))))

;;;; Functions for CrossCat inference
(defn incorporate-column
  "Given a View, a variable name, and a Column GPM, incorporates the Column into the View."
  [view var-name column]
  ;; Introducing a new column means we only need to create the correct number of categories,
  ;; then update the row category assignments within them.
  (update view
          :columns
          assoc var-name (column/update-column column (:latents view))))

(defn unincorporate-column
  "Given a View and a variable name unincorporates the Column from the View."
  [view var-name]
  (update view :columns dissoc var-name))

(defrecord View [columns latents assignments]
  gpm.proto/GPM
  (logpdf [this targets constraints]
    ;; To find the conditional probability of the targets given the constraints,
    ;; we use standard Bayes' Rule.
    ;;                             P(targets, constraints)   <-- joint
    ;; P(targets | constraints) = -------------------------
    ;;                                P(constraints)
    ;; and in the log domain,
    ;;
    ;; logP(targets | constraints) = logP(targets, constraints) - logP(constraints)
    (let [alpha (:alpha latents)
          crp-counts (assoc (:counts latents) :aux alpha)
          n (apply + (vals crp-counts))
          crp-weights (reduce-kv (fn [m k v]
                                   (assoc m k (Math/log (/ v n))))
                                 {}
                                 crp-counts)
          ;; Map of category->loglikelihood.
          ;; We must reweight the logpdf of each category by its CRP weight.
          lls-joint (column-logpdfs columns (merge targets constraints))
          logp-joint (utils/logsumexp (vals (merge-with + lls-joint crp-weights)))
          lls-constraints (column-logpdfs columns constraints)
          logp-constraints (utils/logsumexp (vals (merge-with + lls-constraints crp-weights)))]
      (- logp-joint logp-constraints)))
  (simulate [this targets constraints n-samples]
    (if (nil? targets)
      '()
      (let [crp-counts (:counts latents)
            n (apply + (vals crp-counts))
            alpha (:alpha latents)
            z (+ n alpha)
            crp-weights (reduce-kv (fn [m k v]
                                     (assoc m k (Math/log (/ v z))))
                                   {:aux (Math/log (/ alpha z))}
                                   crp-counts)
            constraint-weights (if (empty? constraints)
                                 {}
                                 ;; If constraints aren't empty, you must re-weight
                                 ;; based on logpdf of constrained columns, across
                                 ;; all their categories.
                                 (->> constraints
                                      (map (fn [[constraint-var constraint-val]]
                                             (let [column (get columns constraint-var)
                                                   aux (column/generate-category column)]
                                               ;; Find the logpdf of the constraint in each of each the categories,
                                               ;; including the auxiliary one introduced by the CRP.
                                               (merge (column/category-logpdfs column {constraint-var constraint-val})
                                                      {:aux (gpm.proto/logpdf aux {constraint-var constraint-val} {})}))))
                                      (apply merge-with + {})))
            unnormalized-weights (merge-with + crp-weights constraint-weights)
            weights (utils/log-normalize unnormalized-weights)
            logps {:p weights}]
        ;; Sample a category assignment, then simulate a value from that category in each of
        ;; the constituent columns.
        ;; It is important to note that each sample is independent, and the underlying View
        ;; is not at all affected for "sequential" simulations.
        (repeatedly n-samples #(let [category-idx (primitives/simulate :log-categorical logps)]
                                 (->> targets
                                      (map (fn [var-name]
                                             {var-name (column/crosscat-simulate (get columns var-name) category-idx)}))
                                      (apply merge)))))))

  gpm.proto/Incorporate
  (incorporate [this values]
    (let [category-key (primitives/crp-simulate-counts {:alpha (:alpha latents)
                                                        :counts (:counts latents)})]
      (incorporate-into-category this values category-key)))
  (unincorporate [this values]
    (let [categories (get-in this [:assignments values])
          category-remove (rand-nth (filter #(not= :row-ids %) (keys categories)))]
      (unincorporate-from-category this values category-remove)))

  gpm.proto/Score
  (logpdf-score [this]
    ;; Within a view, each column is independent, given row-category assignments.
    ;; Therefore the score of the view is just the sum of the score of the
    ;; constituent columns.
    (reduce (fn [acc [_ column]]
              (+ acc (gpm.proto/logpdf-score column)))
            0
            columns)))

(defn construct-view-from-latents
  "Constructor for a View GPM, given a spec for the View, latent
  assignments of data to their respective categories, statistical types of the columns,
  and data. Used in CrossCat inference.

  spec: a View specification, defined as map of {var-name var-hypers}.
  types: the statistical types of the variables contained in the columns (e.g. :bernoulli).
  latents: a map of the below structure, used to keep track of row-category assignments,
           as well as category sufficient statistics:
             {:alpha  number                     The concentration parameter for the Column's CRP
              :counts {category-name count}      Maps category name to size of the category. Updated
                                                 incrementally instead of being calculated on the fly.
              :y {row-identifier category-name}  Maps rows to their current category assignment.
  data: the data belonging to the Column. Must be a map of {row-id {var-name value}},
        including nil values.
  options (optional): Information needed in the column; e.g. For a :categorical Column,
                      `options` would contain a list of possible values the variable could take.
  crosscat (optional): Flag to indicate use in a CrossCat model. This affects how data is handled
                       internally, as CrossCat inference relies on unique row identifiers for
                       efficient inference."
  ([spec latents types data]
   (construct-view-from-latents spec latents types data {:options {}}))
  ([spec latents types data {:keys [options]}]
   (let [hypers (:hypers spec)
         var-names (keys hypers)
         columns (->> var-names
                      ;; For every variable, select only that variable's value
                      ;; for each row of data. A Column GPM is then constructed
                      ;; using this column-specific data.
                      (map (fn [var-name]
                             (let [var-data (reduce-kv (fn [m row-id row-data]
                                                         (assoc m
                                                                row-id
                                                                (get row-data var-name)))
                                                       {}
                                                       data)
                                   var-type (get types var-name)]
                               {var-name (column/construct-column-from-latents
                                          var-name
                                          var-type
                                          (get hypers var-name)
                                          latents
                                          var-data
                                          {:options options})})))
                      (apply merge))
         assignments (reduce-kv (fn [assignments' row-id datum]
                                  (let [y (get-in latents [:y row-id])]
                                    (-> assignments'
                                        (update-in [datum y] (fnil inc 0))
                                        (update-in [datum :row-ids] (fnil #(conj % row-id) #{})))))
                                {}
                                data)]
     (->View columns latents assignments))))
