(ns inferenceql.inference.scripts.clojurecat
  (:require [clojure.tools.cli :refer [parse-opts]]
            [clojure.java.io :as io]
            [clojure.edn :as edn]
            [inferenceql.inference.models.load :as models.load]
            [inferenceql.inference.inference :as infer])
  (:gen-class))

(def cli-options
  ;; Config file that specifies the null-character and variables to ignore.
  [["-c" "--config CONFIG" "path to config .edn, must specify :null-character and :ignore-columns"
    :parse-fn #(-> % slurp edn/read-string)
    :default {:null-character "" :ignore-columns []}
    :validate [#(and (apply every-pred (map (fn [k] (contains? % k))
                                            [:null-character
                                             :ignore-columns
                                             :model
                                             :n-infer-iters]))
                     (or (= (:model %) :xcat)
                         (= (:model %) :dpmm))
                     (pos? (:n-infer-iters %)))
               "must specify :null-character, :ignore-columns, and :model which must be :xcat or :dpmm"]]
   ;; Path to csv to process.
   ["-d" "--data DATA_CSV" "path to .csv data file"
    :validate [#(.exists (io/file %)) "File must exist"]]
   ["-o" "--output OUTPUT" "path for output .edn model file"
    :default "model_out.edn"]
   ;; A boolean option defaulting to nil
   ["-h" "--help"]])

(defn pr-binary-status
  "Given a single argument and function that takes only that argument,
  prints the given message before completing the task and updating the
  printed screen to show that the task is completed."
  [arg function message]
  (print message "... ")
  (let [result (function arg)]
    (print "DONE\n")
    result))

(defn -main [& args]
  (let [parsed (parse-opts args cli-options)
        options (:options parsed)
        config (:config options)
        data (:data options)]
    (println (str "CONFIG: "
                  "\n\tcsv-path:             " data
                  "\n\tnull character:       " (:null-character config)
                  "\n\tignoring variables:   " (:ignore-columns config)
                  "\n\tnumber of inf. iters: " (:n-infer-iters config)
                  "\n\tsave location:        " (:output options)
                  "\n-----------------------------"))
    (-> config
        (assoc :csv-path data)
        (pr-binary-status models.load/load-data "Loading data")
        (pr-binary-status #(-> (models.load/infer-types % (:null-character config))
                               (assoc :data %)
                               (merge config)) "Inferring types")
        (pr-binary-status models.load/init-gpm "Initializing GPM")
        (#(do (println "Performing inference...") %))
        (infer/infer (:n-infer-iters config))
        (pr-binary-status (fn [model]
                            (->> model
                                 (pr-str)
                                 (spit (:output options))))
                          (str "Saving inferred model to: " (:output options))))))
