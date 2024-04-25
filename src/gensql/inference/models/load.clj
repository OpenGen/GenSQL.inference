(ns gensql.inference.models.load
  (:require [clojure.string :as string :refer [lower-case]]
            [clojure.data.csv :as csv]
            [clojure.java.io :as io]
            [clojure.walk :as walk]
            [clojure.edn :as edn]
            [medley.core :as medley]
            [gensql.inference.gpm.view :as view]
            [gensql.inference.gpm.crosscat :as xcat]))

(defn value-coercer
  "Returns a function that will attempt to coerce a value to a data type
  compatible with the given statistical type.
  This function was copied from gensql.query.
  It should be removed when it is available in a separate repo."
  [stattype]
  (let [coerce (case stattype
                  :ignore str
                  :nominal str
                  :numerical (comp double edn/read-string))]
    (fn [value]
      (when-not (string/blank? value)
        (coerce value)))))

(defn row-coercer
  "Returns a function that will attempt to coerce the values in a map to values
  that match on the statistical types provided.
  This function was copied from gensql.query.
  It should be removed when it is available in a separate repo."
  [variable->stattype]
  (fn [row]
    (reduce-kv (fn [row variable stattype]
                 (update row variable (value-coercer stattype)))
               row
               variable->stattype)))

(defn csv-data->maps
  "Given csv data as output from a `data.csv` reader,
  converts each row into a map where the keys are the
  column names specified in the first row of the data."
  [csv-data]
  (map zipmap
       (->> (first csv-data) ;; First row is the header
            repeat)
       (rest csv-data)))

(defn load-data
  "Loads the data contained in the csv at `csv-path`
  Coerces values based on data types in schema."
  [csv-path schema]
  (let [coercer (row-coercer schema)
        rows (-> csv-path
                 (io/reader)
                 (csv/read-csv)
                 (csv-data->maps)
                 (walk/keywordize-keys))]
    (map coercer rows)))

(defn load-schema
  [schema-path]
  (let [schema (-> schema-path slurp edn/read-string walk/keywordize-keys)]
    (medley/remove-vals #{:ignore} schema)))

(defn infer-type
  "Given the set of unique data from a particular column,
  infers the type of the data."
  [data]
  (let [bernoulli? (and (<= (count data) 2)
                        ;; The below condition can be removed if one wishes
                        ;; to consider two-option categorical variables as
                        ;; bernoulli variables. Note that inference conducted
                        ;; over bernoulli variables and two-option categorical
                        ;; variables are not the same, and will likely exhibit
                        ;; different behavior throughout inference.
                        (or (contains? (set (map lower-case data)) "true")
                            (contains? (set (map lower-case data)) "false")))
        all-numbers? (every? number? data)
        ;; The current threshold for delineating numerical from categorical types
        ;; for numerical data is set to 10. This was chosen empirically,
        ;; and can be adjusted accordingly.
        large-set? (>= (count data) 10)]
    (if bernoulli?
      :bernoulli
      (if (and all-numbers? large-set?)
        :gaussian
        :categorical))))

(defn gensql-inference-type
  "Given a `schema` from gensql.structure-learning, returns the gensql.inference type for `variable`."
  [schema variable]
  (let [gensql-type (get schema variable)
        ;; Maps from structure-learning types to gensql.inference types.
        conversion {:numerical :gaussian
                    :nominal :categorical}]
    ;; NOTE: We should not see :ignored columns if gensql.inferece is used via the
    ;; gensql.structure-learning pipeline as it provides ignored.csv, which has ignored
    ;; columns removed.
    (assert (not= :ignored gensql-type))
    (conversion gensql-type)))

(defn options
  "Returns a collection of unique values taken on by `col` in `data`"
  [data col]
  ;; We can assume the null character is "" because when gensql.inference is used
  ;; via gensql.structure-learning, data should have gone through a nullification stage.
  (->> data
       (map #(get % col))
       (remove #(string/blank? (str %)))
       (distinct)))

(defn types-and-options
  "Returns a map of with :types and :options.
    :types - map of column name to gensql.inference type.
    :options - map of column name to unique category options, for categorical columns."
  [schema data]
  (let [col-names (keys (first data))]
    (reduce (fn [acc col]
              (let [var-type (gensql-inference-type schema col)]
                (cond-> (assoc-in acc [:types col] var-type)
                        ;; If the type is categorical, we must record all possible
                        ;; values the col could assume.
                        (= :categorical var-type) (assoc-in [:options col] (options data col)))))
            {:types {} :options {}}
            col-names)))

(defn init-gpm
  [model-type data types-and-opts]
  (let [{:keys [types options]} types-and-opts]
    (case model-type
      :dpmm (view/update-hyper-grids
             (reduce-kv (fn [dpmm row-id datum]
                          (view/incorporate-by-rowid dpmm datum row-id))
                        (view/construct-view-from-types types options)
                        (zipmap (range) data)))
      :xcat (xcat/update-hyper-grids-xcat
             (reduce-kv (fn [xcat row-id datum]
                          (xcat/incorporate-by-rowid xcat datum row-id))
                        (xcat/construct-xcat-from-types types options)
                        (zipmap (range) data)))
      (throw (ex-info (str "model must be either :dpmm or :xcat : " model-type)
                      {:model model-type})))))
