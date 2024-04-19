(ns gensql.inference.gpm.crosscat
  (:require [clojure.set :as set]
            [clojure.edn :as edn]
            [gensql.inference.gpm.conditioned :as conditioned]
            [gensql.inference.gpm.constrained :as constrained]
            [gensql.inference.gpm.view :as view]
            [gensql.inference.gpm.column :as column]
            [gensql.inference.gpm.primitive-gpms :as pgpms]
            [gensql.inference.gpm.proto :as gpm.proto]
            [gensql.inference.primitives :as primitives]))

(defn update-hyper-grids
  "Given a collection of columns, updates the columns' respective hyper grids."
  [columns]
  (reduce-kv (fn [acc col-name col]
               (let [hyper-grid (pgpms/hyper-grid (:stattype col) (vals (:data col)))]
                 (-> acc
                     (assoc col-name (-> col
                                         (assoc :hyper-grid hyper-grid)
                                         (column/update-hypers))))))
             {}
             columns))

(defn update-hyper-grids-xcat
  "Given a collection of columns, updates the columns' respective hyper grids."
  [xcat]
  (update xcat :views (fn [views]
                        (reduce-kv (fn [acc view-name view]
                                     (assoc acc view-name (update view
                                                                  :columns
                                                                  update-hyper-grids)))
                                   {}
                                   views))))

(defn get-data
  "Grabs data from xcat by row-id."
  [xcat row-id]
  (reduce-kv (fn [x _ view]
               (merge x (view/get-data view row-id)))
             {}
             (:views xcat)))

(defn calculate-weights
  "Given an XCat GPM, for each row calculates the logpdf of the data contained in each row.
  Used for search in ensembles."
  [xcat]
  (let [row-ids (-> xcat :views first second :latents :y keys)]
    (reduce (fn [m row-id]
              (let [row-data (into {} (filter (comp some? val) (get-data xcat row-id)))]
                (assoc m row-id (gpm.proto/logpdf xcat row-data {}))))
            {}
            row-ids)))

(defn generate-view-latents
  "Given a CrossCat model, samples view assignments parameterized by the
  current concentration parameter value."
  [n alpha]
  (let [[_ assignments] (primitives/crp-simulate n {:alpha alpha})]
    (zipmap (range) (shuffle assignments))))

(defn incorporate-by-rowid
  "Given an XCat GPM, a map of values, and a row-id, incorporates
  the values into the GPM under the given row-id."
  [xcat values row-id]
  (-> xcat
      ((fn [gpm]
         ;; Incorporate correct variables within the datumn into their respective views.
         (reduce-kv (fn [m view-idx view]
                      (let [view-vars (keys (:columns view))
                            x-view (select-keys values view-vars)]
                        (update-in m
                                   [:views view-idx]
                                   #(-> (view/incorporate-by-rowid % x-view row-id)))))
                    gpm
                    (:views xcat))))))

(defrecord XCat [views latents]
  gpm.proto/GPM
  (logpdf [_ targets constraints]
    (let [intersection (set/intersection (set (keys targets)) (set (keys constraints)))]
      ;; If targets are the same as constraints, the logpdf is 0.
      (cond
        (= targets constraints)
        0.0
        ;; If the targets and constraints are not equal but the overlapping parts are,
        ;; just remove the overlapping keys and recur the scores. 
        (every? (fn [shared-key]
                  (= (get targets shared-key)
                     (get constraints shared-key)))
                intersection)
        (reduce-kv (fn [logp _ view]
                     ;; Filtering view variables happens naturally.
                     (let [view-logp (gpm.proto/logpdf view
                                                       (apply dissoc targets intersection)
                                                       constraints)]
                       (+ logp view-logp)))
                   0.0
                   views)
        ;; If the intersection keys map to different values, the score is -Inf.
        :else ##-Inf)))
  (simulate [_ targets constraints]
    ;; Catch overlap of targets and constraints and assure constraint is sampled. 
    (let [intersection (set/intersection (set targets) (set (keys constraints)))
          unconstrained-targets (vec (remove intersection (set targets)))]
        (->> views
             (map (fn [[_ view]]
                    (gpm.proto/simulate view unconstrained-targets constraints)))
             (filter not-empty)
             (apply merge (select-keys constraints intersection)))))
  gpm.proto/Incorporate
  (incorporate [this x]
    (let [row-id (gensym)]
      (incorporate-by-rowid this x row-id)))
  (unincorporate [this x]
    (-> this
        ((fn [xcat]
           ;; Unincorporate correct variables within the datumn from one of their associated views.
           (reduce-kv (fn [m view-name view]
                        (let [view-vars (keys (:columns view))
                              x-view (select-keys x view-vars)]
                          (update-in m [:views view-name] #(-> (gpm.proto/unincorporate % x-view)
                                                               (update :columns update-hyper-grids)))))
                      xcat
                      views)))))
  gpm.proto/Score
  (logpdf-score [_]
    (reduce (fn [acc [_ view]]
              (+ acc (gpm.proto/logpdf-score view)))
            0
            views))
  gpm.proto/Variables
  (variables [{:keys [views]}]
    (into #{}
          (mapcat gpm.proto/variables)
          (vals views)))
  gpm.proto/Condition
  (condition [this conditions]
    (conditioned/condition this conditions))
  gpm.proto/Constrain
  (constrain [this event opts]
    (constrained/constrain this event opts)))

(defn incorporate-column
  "Incorporates a column in to the model at the specified view."
  [xcat column view-assignment]
  (let [var-name (:var-name column)]
    (-> xcat
        ;; Update sufficient statistics of the XCat CRP.
        (assoc-in [:latents :z var-name] view-assignment)
        (update-in [:latents :counts view-assignment] inc)
        ;; Incorporate the column to the correct view.
        (update-in [:views view-assignment] #(view/incorporate-column % column)))))

(defn unincorporate-column
  "Unincorporates a column from the model with the specified name."
  [xcat var-name]
  (let [z (-> xcat :latents :z (get var-name))]
    (-> xcat
        ;; Update sufficient statistics of the XCat CRP.
        (update-in [:latents :z] dissoc var-name)
        (update-in [:latents :counts z] dec)
        ;; Unincorporate the column from the associated view.
        (update-in [:views z] #(view/unincorporate-column % var-name)))))

(defn filter-empty-views
  "Filters empty views from a CrossCat model."
  [xcat]
  (let [latents (:latents xcat)
        views-to-remove (keys (filter #(zero? (second %)) (:counts latents)))]
    (-> xcat
        (update :views #(apply dissoc % views-to-remove))
        (update-in [:latents :counts] #(apply dissoc % views-to-remove)))))

(defn view-logpdf-scores
  "Given an XCat GPM and a Column, calculates the logpdf-score of the Column GPM
  if it were to be incorporated into each of the constituent View GPMs."
  [xcat column]
  (reduce-kv (fn [scores view-name view]
               (let [view-latents (:latents view)]
                 (assoc scores
                        view-name
                        (gpm.proto/logpdf-score (column/update-column column view-latents)))))
             {}
             (:views xcat)))

(defn generate-view
  "Generates an empty view with latent assignments generated from a CRP
  with concentration parameter alpha."
  [row-ids]
  (let [n (count row-ids)
        alpha (primitives/simulate :gamma {:k 1 :theta 1})
        [table-counts assignments] (primitives/crp-simulate n {:alpha alpha})
        counts (zipmap (range) table-counts)
        y (zipmap row-ids assignments)
        latents {:alpha alpha
                 :counts counts
                 :y y}
        columns {}
        assignments {}]
    (view/->View columns latents assignments)))

(defn add-aux-views
  "Add m auxiliary Views to the given XCat GPM."
  [xcat m]
  (let [row-ids (-> xcat :views first second :latents :y keys)]
    (reduce (fn [xcat' _]
              (let [view-symbol (gensym)]
                (-> xcat'
                    (assoc-in [:latents :counts view-symbol] 0)
                    (assoc-in [:views view-symbol] (generate-view row-ids)))))
            xcat
            (range m))))

(defn construct-xcat-from-latents
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
                      `options` would contain a list of possible values the variable could take."
  ([spec latents data]
   (construct-xcat-from-latents spec latents data {:options {}}))
  ([spec latents data {:keys [options]}]
   (let [views (:views spec)
         types (:types spec)
         global-latents (:global latents)
         ;; Create views with correctly populated categories.
         views' (reduce-kv (fn [acc view-name view]
                             (let [view-vars (-> view :hypers keys)]
                               (assoc acc view-name (view/construct-view-from-latents
                                                     view
                                                     (get-in latents [:local view-name])
                                                     types
                                                     ;; Need to filter each datum for view-specific
                                                     ;; variables in order to avoid error.
                                                     (reduce-kv (fn [data' row-id datum]
                                                                  (assoc data' row-id (select-keys datum view-vars)))
                                                                {}
                                                                data)
                                                     {:options options
                                                      :crosscat true}))))
                           {}
                           views)
         ;; Create unordered (bag) for latent counts and column view assignments.
         xcat-latents (reduce-kv (fn [m view-idx _]
                                   (let [var-names (keys (:hypers (get views view-idx)))]
                                     (-> m
                                         (assoc-in [:counts view-idx] (count var-names))
                                         (update :z #(reduce (fn [acc var-name]
                                                               (assoc acc var-name view-idx))
                                                             %
                                                             var-names)))))
                                 {:alpha (:alpha global-latents)
                                  :counts {}
                                  :z {}}
                                 views')]
     (->XCat views' xcat-latents))))

(defn construct-xcat-from-hypers
  "Constructor of a XCat GPM, given a specification for variable hyperparameters, as well
  as variable statistical types."
  [spec]
  (let [latents {:global {:alpha 1 :counts {} :z {}}
                 :local {0 {:alpha 1
                            :counts {}
                            :y {}}}}
        options (:options spec)
        data {}]
    (construct-xcat-from-latents spec latents data {:options options})))

(defn construct-xcat-from-types
  "Constructor of a XCat GPM, given a specification for variable types, wherein all
  variables are placed into the same view initially."
  [types options]
  (let [hypers (reduce-kv (fn [hypers' var-name var-type]
                            (assoc hypers' var-name (case var-type
                                                      :bernoulli {:alpha 0.5 :beta 0.5}
                                                      :categorical {:alpha 1}
                                                      :gaussian {:m 0 :r 1 :s 1 :nu 1})))
                          {}
                          types)
        spec {:views {0 {:hypers hypers}}
              :types types
              :options options}]
    (construct-xcat-from-hypers spec)))

(defn xcat->mmix
  "Converts a specified XCat GPM into a Multimixture spec."
  [xcat]
  (let [view-number (fn [[view-kw _view]]
                      (->> (name view-kw)
                           (re-matches #"view_(\d+)")
                           second
                           edn/read-string))
        ;; Sort views by view-number (asc) and then collect just views.
        views (map second (sort-by view-number (:views xcat)))
        [vars views]
        (reduce (fn [[vars views] view]
                  ;; For each view, record the type of each column,
                  ;; and convert each category into a Multimixture spec representation.
                  (let [view-latents (:latents view)
                        view-counts (:counts view-latents)
                        view-variables (reduce-kv (fn [var-types col-name column]
                                                    (assoc var-types col-name (:stattype column)))
                                                  {}
                                                  (:columns view))
                        z (reduce + (vals view-counts))
                        cat-number (fn [cat-kw]
                                     (->> (name cat-kw)
                                          (re-matches #"cluster_(\d+)")
                                          second
                                          edn/read-string))
                        category-names (sort-by cat-number (keys view-counts))
                        categories (reduce (fn [categories category-name]
                                             (let [;; The prior of the category is proportional to its size.
                                                   category-weight (double (/ (get view-counts category-name) z))
                                                   category
                                                   (reduce-kv (fn [column-categories _ column]
                                                                (let [;; If there is no category for a given column, this means
                                                                      ;; that there is no associated data with that column in the rows within
                                                                      ;; that category. Because the types are collapsed, we can generate
                                                                      ;; a new (empty) category for that column.
                                                                      col-cat (get-in column [:categories category-name] (column/generate-category column))
                                                                      col-stattype (:stattype column)]
                                                                  (merge column-categories
                                                                         (pgpms/export-category col-stattype col-cat))))
                                                              {}
                                                              (:columns view))]
                                               (conj categories {:probability category-weight
                                                                 :parameters category})))
                                           []
                                           category-names)]
                    [(merge vars view-variables)
                     (conj views categories)]))
                [{} []]
                views)]
    {:vars vars
     :views views}))

(defn xcat?
  "Checks if the given GPM is an XCat GPM."
  [stattype]
  (and (record? stattype)
       (instance? XCat stattype)))
