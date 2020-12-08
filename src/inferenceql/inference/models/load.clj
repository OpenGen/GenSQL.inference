(ns inferenceql.inference.models.load
  (:require [clojure.string :refer [lower-case]]
            [clojure.data.csv :as csv]
            [clojure.java.io :as io]
            [clojure.walk :as walk]
            [inferenceql.inference.gpm.view :as view]
            [inferenceql.inference.gpm.crosscat :as xcat]))

(defn csv-data->maps
  "Given csv data as output from a `data.csv` reader,
  converts each row into a map where the keys are the
  column names specified in the first row of the data."
  [csv-data]
  (map zipmap
       (->> (first csv-data) ;; First row is the header
            repeat)
       (rest csv-data)))

(defn clean-data
  "Given a collection of maps representing separate data,
  and a specified null character, safely parses the data
  into numerical and categorical types."
  [data null-character]
  (reduce (fn [data' datum]
            (conj data' (reduce-kv (fn [m k v]
                                     (if (= v null-character)
                                       (assoc m k nil)
                                       ;; If there is a parsing error, leave the value as is
                                       ;; (very likely a quasi-numerical string that Clojure
                                       ;; attempted to turn into a number).
                                       ;; NOTE: using `read-string` instead of `edn/read-string`
                                       ;; leads to inference speed increases by about 15%.
                                       ;; Cause is currently unknown.
                                       (let [v' (try (read-string (lower-case v))
                                                     (catch Exception _ v))]
                                         (if (symbol? v')
                                           (assoc m k v)
                                           (assoc m k v')))))
                                   {}
                                   datum)))
          []
          data))

(defn filter-variables
  "Given data and a non-zero number of variable names,
  removes all of the specified variables from each datum in data."
  [data & variables]
  (map #(apply dissoc % variables) data))

(defn load-data
  "Loads the data contained in the csv at `csv-path`
  and filters out `ignore-columns`, noting which character
  represents empty values."
  [{:keys [csv-path null-character ignore-columns]}]
  (-> csv-path
      (io/reader)
      (csv/read-csv)
      (csv-data->maps)
      (clean-data null-character)
      (#(apply filter-variables % ignore-columns))
      (walk/keywordize-keys)))

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

(defn infer-types
  "Given a collection of data, where each datum is a map with variable names as keys
  and the respective variable data as values, and a specified null character,
  infers the statistical type of each column in the data."
  [data null-character]
  (let [variables (keys (first data))]
    (reduce (fn [types variable]
              (let [;; Gather all unique non-nil values from the current column.
                    var-data (set (filter #(and (not= null-character %)
                                                (some? %)
                                                (not= (symbol null-character) %))
                                          (map #(get % variable) data)))
                    var-type (infer-type var-data)
                    ;; If the type is categorical, we must record all possible
                    ;; values the variable could assume.
                    options (if (= :categorical var-type)
                              (into [] var-data)
                              nil)]
                (-> types
                    (assoc-in [:types variable] var-type)
                    (#(if (= :categorical var-type)
                        (assoc-in % [:options variable] options)
                        %)))))
            {:types {} :options {}}
            variables)))

(defn init-gpm
  [{:keys [model types options data]}]
  (case model
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
    (throw (ex-info (str "model must be either :dpmm or :xcat : " model)
                    {:model model}))))
