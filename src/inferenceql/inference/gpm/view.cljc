(ns inferenceql.inference.gpm.view
  "Implementation of a GPM that represents a population of data of potentially
  different primitive types, but the same row categorizations. For a tabular dataset,
  this GPM abstracts a subset of columns of that dataset, the rows of which are clustered
  together in categories. This can also be considered as Dirichlet Process Mixture Model (DPMM).
  See `inferenceql.inference.gpm/view` for details."
  (:require [inferenceql.inference.utils :as utils]
            [inferenceql.inference.primitives :as primitives]
            [inferenceql.inference.gpm.column :as column]
            [inferenceql.inference.gpm.proto :as gpm.proto]
            [clojure.set]))

(defn column-logpdfs
  "Given a map of columns, and targets, returns a map of category probabilities of the targets."
  ([columns targets]
   (column-logpdfs columns targets {:add-aux false}))
  ([columns targets {:keys [add-aux]}]
   (->> targets
        (map (fn [[var-name target]]
               (column/category-logpdfs (get columns var-name) {var-name target} {:add-aux add-aux})))
        (apply merge-with + {}))))

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
  ([view values category-key]
   (incorporate-into-category view values category-key (gensym)))
  ([view values category-key row-id]
   (-> view
       ;; latents are used only in CrossCat inference. This implementation of row assignments
       ;; may be deprecated, depending on how the the inference procedures are adjusted.
       (assoc-in [:latents :y row-id] category-key)
       (update-in [:latents :counts category-key] (fnil inc 0))
       ;; When updating assignments, we want to
       ;; 1.) Increase the category count for the given set of values.
       (update-in [:assignments values :categories category-key] (fnil inc 0))
       ;; 2.) Add the row-id to the set of row-ids associated with the given values.
       (update-in [:assignments values :row-ids] (fnil #(conj % row-id) #{}))
       ;; 3.) Update the relevant columns by assoc'ing the value associated with the row-id.
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
       ;; When updating assignments, we want to
       ;; 1.) Decrease the current category count for the given set of values.
       (update-in [:assignments values :categories category-remove] dec)
       ;; 2.) Remove the category count if it hits zero as a result.
       (update :assignments (fn [assignments]
                              (if (zero? (get-in assignments [values :categories category-remove]))
                                (update-in assignments [values :categories] dissoc category-remove)
                                assignments)))
       ;; 3.) Disj the row-id from the set of row-ids for the given set of values.
       (update-in [:assignments values :row-ids] #(disj % row-id-remove))
       ;; 4.) Dissoc the given set of values if there are no longer rows associated with it.
       (update :assignments (fn [assignments]
                              (if (empty? (get-in assignments [values :row-ids]))
                                (dissoc assignments values)
                                assignments)))
       ;; 5.) Update the relevant columns by dissoc'ing the value associated with the row-id.
       (update :columns #(reduce-kv (fn [columns' col-name column]
                                      (assoc columns' col-name (column/crosscat-unincorporate
                                                                column
                                                                category-remove
                                                                row-id-remove)))
                                    {}
                                    %)))))

(defn incorporate-by-rowid
  [gpm values row-id]
  (let [category-key (primitives/crp-simulate-counts {:alpha (-> gpm :latents :alpha)
                                                      :counts (-> gpm :latents :counts)})]
    (incorporate-into-category gpm values category-key row-id)))

(defn update-hyper-grids
  "doc-string"
  [view]
  (update view :columns #(reduce-kv (fn [columns' col-name column]
                                      (assoc columns' col-name (column/update-hyper-grid column)))
                                    {}
                                    %)))

;;;; Functions for CrossCat inference
(defn incorporate-column
  "Given a View, a variable name, and a Column GPM, incorporates the Column into the View."
  [view column]
  ;; Introducing a new column means we only need to create the correct number of categories,
  ;; then update the row category assignments within them.
  (let [var-name (:var-name column)
        col-data (:data column)]
    (-> view
        ;; Adjust the column data to a particular set of latents.
        (assoc-in [:columns var-name] (column/update-column column (:latents view)))
        ;; Update the assignments attribute, which maps values to their category
        ;; assignments and associated row-ids used in CrossCat inference.
        (update :assignments (fn [assignments]
                               (if (empty? assignments)
                                 ;; If the View is empty, we just need to adjust assignments
                                 ;; according to the column's data.
                                 (reduce-kv (fn [assignments' row-id datum]
                                              (let [datum' {var-name datum}
                                                    y (get-in (:latents view) [:y row-id])]
                                                (-> assignments'
                                                    (update-in [datum' :categories y] (fnil inc 0))
                                                    (update-in [datum' :row-ids] (fnil #(conj % row-id) #{})))))
                                            {}
                                            col-data)
                                 ;; If the View is not empty, some care must be had in terms
                                 ;; of updating assignments.
                                 ;; An example would be merging the below assignments map
                                 ;;   {{:var-1 :val-1} {:categories {:one 1}
                                 ;;                     :row-ids #{1 2 3}}}
                                 ;; with the new column-data
                                 ;;   {1 :val-2
                                 ;;    2 :val-3
                                 ;;    3 :val-2}
                                 ;; to update assignments as
                                 ;;   {{:var-1 :val-1
                                 ;;     :var-2 :val-2} {:categories {:one 1}
                                 ;;                     :row-ids #{1 3}}
                                 ;;    {:var-1 :val-1
                                 ;;     :var-2 :val-3} {:categories {:one 1}
                                 ;;                     :row-ids #{2}}}
                                 (reduce-kv (fn [m k v]
                                              (let [row-ids (:row-ids v)]
                                                (reduce (fn [m' row-id]
                                                          (let [row-data (get col-data row-id)
                                                                k' (assoc k var-name row-data)
                                                                y (get-in (:latents view) [:y row-id])]
                                                            (-> m'
                                                                (update-in [k' :categories y] (fnil inc 0))
                                                                (update-in [k' :row-ids] (fnil #(conj % row-id) #{})))))
                                                        m
                                                        row-ids)))
                                            {}
                                            assignments)))))))

(defn unincorporate-column
  "Given a View and a variable name unincorporates the Column from the View."
  [view var-name]
  (let [singleton? (= 1 (count (:columns view)))
        view' (update view :columns dissoc var-name)]
    ;; If we're unincorporating a View GPM's only column,
    ;; it is simplest to clear the assignments attribute.
    (if singleton?
      (assoc view' :assignments {})
      (update view' :assignments (fn [assignments]
                                   ;; For every assignment to category mapping,
                                   ;; remove the current variable from each key.
                                   (let [assignments' (map (fn [[k v]] {(dissoc k var-name) v})
                                                           assignments)
                                         merge-fn (fn [a b]
                                                    (assert (= (keys a) (keys b)))
                                                    {:categories (merge-with + (:categories a) (:categories b))
                                                     :row-ids (clojure.set/union (:row-ids a) (:row-ids b))})]
                                     (apply merge-with merge-fn assignments')))))))

(defrecord View [columns latents assignments]
  gpm.proto/GPM
  (logpdf [this targets constraints]
    (let [modeled? (set (keys columns))
          ;; Filtering targets and constaints for only those which are modeled.
          targets (into {} (filter #(modeled? (key %)) targets))
          constraints (into {} (filter #(modeled? (key %)) constraints))

          ;; To find the conditional probability of the targets given the constraints,
          ;; we use standard Bayes' Rule.
          ;;                             P(targets, constraints)   <-- joint
          ;; P(targets | constraints) = -------------------------
          ;;                                P(constraints)
          ;; and in the log domain,
          ;;
          ;; logP(targets | constraints) = logP(targets, constraints) - logP(constraints)
          alpha (:alpha latents)
          crp-counts (assoc (:counts latents) :aux alpha)
          n (apply + (vals crp-counts))
          crp-weights (reduce-kv (fn [m k v]
                                   (assoc m k (Math/log (/ v n))))
                                 {}
                                 crp-counts)
          ;; Map of category->loglikelihood.
          ;; We must reweight the logpdf of each category by its CRP weight.
          lls-joint (column-logpdfs columns (merge targets constraints) {:add-aux true})
          logp-joint (utils/logsumexp (vals (merge-with + lls-joint crp-weights)))
          lls-constraints (column-logpdfs columns constraints {:add-aux true})
          logp-constraints (utils/logsumexp (vals (merge-with + lls-constraints crp-weights)))]
      (- logp-joint logp-constraints)))
  (simulate [this targets constraints]
    (let [modeled? (set (keys columns))
          ;; Filtering targets and constaints for only those which are modeled.
          targets (filter modeled? targets)
          constraints (into {} (filter #(modeled? (key %)) constraints))]
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
              logps {:p weights}
              ;; Sample a category assignment, then simulate a value from that category in each of
              ;; the constituent columns.
              category-idx (primitives/simulate :log-categorical logps)]
          (->> targets
               (map (fn [var-name]
                      {var-name (column/crosscat-simulate (get columns var-name) category-idx)}))
               (apply merge))))))

  gpm.proto/Incorporate
  (incorporate [this values]
    (let [category-key (primitives/crp-simulate-counts {:alpha (:alpha latents)
                                                        :counts (:counts latents)})]
      (incorporate-into-category this values category-key)))
  (unincorporate [this values]
    (let [categories (get-in this [:assignments values :categories])
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
   (construct-view-from-latents spec latents types data {:options {} :crosscat false}))
  ([spec latents types data {:keys [options crosscat]}]
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
                                          {:options options
                                           :crosscat crosscat})})))
                      (apply merge))
         assignments (reduce-kv (fn [assignments' row-id datum]
                                  (let [y (get-in latents [:y row-id])]
                                    (-> assignments'
                                        (update-in [datum :categories y] (fnil inc 0))
                                        (update-in [datum :row-ids] (fnil #(conj % row-id) #{})))))
                                {}
                                data)]
     (->View columns latents assignments))))

(defn construct-view-from-hypers
  "Constructor of a View GPM, given a specification for variable hyperparameters, as well
  as variable statistical types."
  [spec types]
  (let [latents {:alpha 1 :counts {} :y {}}
        options (:options spec)
        data {}]
    (construct-view-from-latents spec latents types data {:options options})))

(defn construct-view-from-types
  "Constructor of a View GPM, given a specification for variable types."
  [types options]
  (let [hypers (reduce-kv (fn [hypers' var-name var-type]
                            (assoc hypers' var-name (case var-type
                                                      :bernoulli {:alpha 0.5 :beta 0.5}
                                                      :categorical {:alpha 1}
                                                      :gaussian {:m 0 :r 1 :s 1 :nu 1})))
                          {}
                          types)
        spec {:hypers hypers :options options}]
    (construct-view-from-hypers spec types)))

(defn view?
  "Checks if the given GPM is a View."
  [stattype]
  (and (record? stattype)
       (instance? View stattype)))
