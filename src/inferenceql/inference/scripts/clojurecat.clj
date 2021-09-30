(ns inferenceql.inference.scripts.clojurecat
  (:require [clojure.tools.cli :refer [parse-opts]]
            [clojure.java.io :as io]
            [clojure.edn :as edn]
            [inferenceql.inference.models.load :as models.load]
            [inferenceql.inference.inference :as infer])
  (:gen-class))

(def cli-options
  [["-c" "--config CONFIG" "path to config .edn, must specify :model"
    :parse-fn #(-> % slurp edn/read-string)
    :validate [#(and (apply every-pred (map (fn [k] (contains? % k))
                                            [:model
                                             :n-infer-iters]))
                     (or (= (:model %) :xcat)
                         (= (:model %) :dpmm)))
               "must specify :model which must be :xcat or :dpmm"]]
   ;; Path to csv to process.
   ["-d" "--data DATA_CSV" "path to .csv data file"
    :validate [#(.exists (io/file %)) "File must exist"]]
   ;; Path to schema for the dataset.
   ["-s" "--schema SCHEMA" "path to .edn schema file"
    :validate [#(.exists (io/file %)) "File must exist"]]
   ["-o" "--output OUTPUT" "path for output .edn model file"
    :default "model_out.edn"]
   ;; A boolean option defaulting to nil
   ["-h" "--help"]])

(defmacro pr-binary-status
  "Executes the expression `form`, but prints `message` beforehand, and prints DONE afterwards.
  Returns the value obtained by executing `form`.
  This macro is useful for displaying execution status to the user on the commandline."
  [form message]
  (let [result-sym (gensym "result")] ; A unique symbol to prevent shadowing.
    `(do
       (print (str ~message "..."))
       (let [~result-sym ~form]
         (print " DONE\n")
         ~result-sym))))

(defn -main [& args]
  (let [{:keys [options]} (parse-opts args cli-options)
        data-path (get options :data)
        schema-path (get options :schema)
        output-path (get options :output)
        model-type (get-in options [:config :model])
        n-infer-iters (get-in options [:config :n-infer-iters])

        ;; Start performing work and updating user.
        schema (pr-binary-status (models.load/load-schema schema-path)
                                 "Loading schema")
        rows (pr-binary-status (models.load/load-data data-path schema)
                               "Loading data")
        types-and-opts (pr-binary-status (models.load/types-and-options schema rows)
                                         "Gathering category options")
        model (pr-binary-status (models.load/init-gpm model-type rows types-and-opts)
                                "Initializing GPM")]
    (println "Performing inference...")
    (infer/infer model n-infer-iters)
    ;; Save the model.
    (pr-binary-status (->> model
                           (pr-str)
                           (spit output-path))
                      (str "Saving inferred model to: " output-path))))

