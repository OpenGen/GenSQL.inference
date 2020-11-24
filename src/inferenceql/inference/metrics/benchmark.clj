(ns inferenceql.inference.metrics.benchmark
  (:require [clojure.java.io :as io]
            [clojure.data.json :as json]
            [clojure.set]
            [clojure.edn :as edn]
            [inferenceql.inference.utils :as utils]
            [inferenceql.inference.gpm.crosscat :as xcat]
            [inferenceql.inference.gpm.view :as view]
            [inferenceql.inference.gpm.column :as column]
            [inferenceql.inference.gpm.primitive-gpms.bernoulli :as bernoulli]
            [inferenceql.inference.gpm.primitive-gpms.categorical :as categorical]
            [inferenceql.inference.gpm.primitive-gpms.gaussian :as gaussian]))

(defn dpmm-gen-fn
  "Given types, options, and training data for a DPMM model,
  returns a generating function that creates an View GPM
  with all rows of the data incorporated into the model."
  [{:keys [types options data]}]
  (let [view-empty (view/construct-view-from-types
                    types
                    options)]
    #(reduce-kv (fn [view row-id row]
                  (view/incorporate-by-rowid view row row-id))
                view-empty
                (zipmap (range) data))))

(defn xcat-gen-fn
  "Given types, options, and training data for an XCat model,
  returns a generating function that creates an XCat GPM
  with all rows of the data incorporated into the model."
  [{:keys [types options data]}]
  (let [xcat-empty (xcat/construct-xcat-from-types
                    types
                    options)]
    #(reduce-kv (fn [xcat row-id row]
                  (xcat/incorporate-by-rowid xcat row row-id))
                xcat-empty
                (zipmap (range) data))))

(defmacro time-data
  "Returns time and data of an expression of the form `(time expression)`
  Returns a map with keys `:result` and `:time`.
  Returns `:time` in milliseconds.
  If the expression results in a lazy sequence, you must pass
  `(time (doall expression))` instead."
  [& body]
  `(let [s# (new java.io.StringWriter)]
     (binding [*out* s#]
       (let [r# (time ~@body)]
         {:result r#
          :time (Float/parseFloat (nth (split (str s#) #" ")
                                       2))}))))

(defn time-execution
  "Given a function `f`, times and executes the function `n` times, returning a vector
  of the respective times."
  [f n]
  (->> #(time-data (f))
       (repeatedly n)
       (map :time)))

(defn microbenchmark
  "Given a function `f`, times and executes the function `n` times, returning a map
  containing the mean and standard deviation of all the execution times."
  [f n]
  (let [times (time-execution f n)]
    (-> {}
        (assoc :mean (utils/average times))
        (assoc :std (utils/std times)))))

(defn generate-query
  "Generates a query at random, given data, its column types, and the number
  of targets and constraints.

  Queries are defined for the various statistical types as follows:
    - bernoulli: randomly choose `true` or `false`
    - categorical: randomly choose from the variable's domain choice set
    - gaussian: sample uniformly from the range [min(data), max(data)]
  "
  [{:keys [types options data n-targets n-constraints simulate?]}]
  (let [;; Randomly choose variables from the model.
        all-vars (into {} (take (+ n-targets n-constraints) (shuffle (vec types))))
        ;; Associate a random value within the domain for each variable.
        all-vals (reduce-kv (fn [acc target target-type]
                              (let [target-val (case target-type
                                                 :bernoulli (rand-nth [true false])
                                                 :categorical (rand-nth (get options target))
                                                 :gaussian (let [data-min (apply min (filter some? (map #(get % target) data)))
                                                                 data-max (apply max (filter some? (map #(get % target) data)))]
                                                             (+ data-min (* (rand) (- data-max data-min))))
                                                 (throw (ex-info (str "not a type: " target-type) {:target-type target-type})))]
                                (assoc acc target target-val)))
                            {}
                            all-vars)
        var-list (keys all-vars)
        ;; Filter targets from all random values.
        targets (select-keys all-vals (take n-targets var-list))
        ;; If a simulate query, we only need the keys, not the random associated values.
        targets (if simulate? (keys targets) targets)
        ;; Filter constraints from remaining random values, ensuring no overlap between targets and constraints.
        constraints (select-keys all-vals (->> var-list (drop n-targets) (take n-constraints)))]
    {:targets targets
     :constraints constraints}))

(defn generate-predicate
  "Generates a predicate at random, given data, its column types, and the number
  of targets and constraints. Predicates are functions that take a single argument
  and checks the argument against a specified condition, returning `true` is the condition
  holds and `false` otherwise.

  Predicates are defined for the various statistical types as follows:
    - bernoulli: function wherein we check equality with randomly chosen `true` or `false`
    - categorical: function wherein we check membership within a random subset of the variable's domain choice set,
                   where the size of the subset is equal to approximately half of the total domain
    - gaussian: function wherein we sample a midpoint uniformly at random from the range [min(data), max(data)],
                and then check whether the argument lies above or below the midpoint (direction chosen at random).
  "
  [{:keys [types options data n-targets n-constraints]}]
  (let [;; Randomly choose variables from the model.
        all-vars (into {} (take (+ n-targets n-constraints) (shuffle (vec types))))
        ;; Associate a random predicate function for each variable.
        all-predicates (reduce-kv (fn [acc target target-type]
                                    (let [predicate (case target-type
                                                      :bernoulli (fn [sample] (= sample (rand-nth [true false])))
                                                      :categorical (let [target-options (get options target)
                                                                         n-options (count target-options)
                                                                         sample-set-size (int (/ n-options 2))
                                                                         sample-set (set (take sample-set-size (shuffle target-options)))]
                                                                     (fn [sample]
                                                                       (contains? sample-set (get sample target))))
                                                      :gaussian (let [data-min (apply min (filter some? (map #(get % target) data)))
                                                                      data-max (apply max (filter some? (map #(get % target) data)))
                                                                      midpoint (+ data-min (* (rand) (- data-max data-min)))
                                                                      gt-lt (if (> (rand) 0.5) > <)]
                                                                  (fn [sample]
                                                                    (gt-lt midpoint (get sample target))))
                                                      (throw (ex-info (str "not a type: " target-type) {:target-type target-type})))]
                                      (assoc acc target predicate)))
                                  {}
                                  all-vars)
        var-list (keys all-vars)
        ;; Filter predicate (constrained) variables.
        predicate-keys (take n-constraints var-list)
        ;; Create function that checks predicate functions for all predicate variables.
        composite-predicate (apply every-pred (vals (select-keys all-predicates predicate-keys)))
        ;; We must join the target variables for the predicate variables because they must all
        ;; be simulated in a given GPM.
        targets (apply conj
                       predicate-keys
                       (keys (select-keys all-predicates (->> var-list (drop n-constraints) (take n-targets)))))
        ;; Create function that checks predicate functions for all target variables.
        target-predicate (apply every-pred (vals (select-keys all-predicates targets)))]
    {:targets targets
     :constrained predicate-keys
     :target-predicate target-predicate
     :constraint-predicate composite-predicate}))

(defn load-model
  "Given a .edn file containing a GPM, loads the GPM into ClojureCat."
  [file]
  (edn/read-string {:readers {'inferenceql.inference.gpm.crosscat.XCat xcat/map->XCat
                              'inferenceql.inference.gpm.view.View view/map->View
                              'inferenceql.inference.gpm.column.Column column/map->Column
                              'inferenceql.inference.gpm.primitive_gpms.bernoulli.Bernoulli bernoulli/map->Bernoulli
                              'inferenceql.inference.gpm.primitive_gpms.categorical.Categorical categorical/map->Categorical
                              'inferenceql.inference.gpm.primitive_gpms.gaussian.Gaussian gaussian/map->Gaussian}}
                   (slurp file)))

(defn load-ensemble
  "Given a directory containing .edn files, each with a different GPM
  of the same type for the same data, loads the ensemble into ClojureCat."
  [dir]
  (let [files (rest (file-seq dir))]
    (mapv (comp load-model #(.getAbsolutePath %)) files)))

(defn write-ensemble
  "Given a file base destination, writes an ensemble of GPMs to file, with
  each GPM in a separate enumerated file."
  [gpm-file-base gpm-ensemble]
  (reduce-kv (fn [_ gpm-number gpm]
               (let [file (str gpm-file-base gpm-number ".edn")]
                 (io/make-parents file)
                 (spit file (pr-str gpm))))
             []
             (zipmap (range) gpm-ensemble)))

(defn get-split
  "Returns a roughly even split of the targets, where the sum of the split is equal to n-labels.
  e.g. targets = [true false], n-labels = 4
       => {true 2 false 2}
  e.g. targets = [:a :b :c], n-labels = 5
       => {:a 1 :b 1 :c 3}"
  [n-labels targets]
  (if (zero? n-labels)
    (zipmap targets (repeat 0))
    (let [init-split (first (reduce (fn [[split' remainder] target]
                                      (let [step (int (/ n-labels (count targets)))
                                            remainder' (- remainder step)
                                            finish? (< remainder' step)
                                            to-add (if finish? (+ step remainder') step)]
                                        [(assoc split' target to-add)
                                         remainder']))
                                    [{} n-labels]
                                    targets))
          remainder (- n-labels (reduce + (vals init-split)))]
      (if (pos? remainder)
        (update init-split (last targets) #(+ % remainder))
        init-split))))

(defn get-sparse-labels
  "Given a vector of labels, returns either the first `n-labels`,
  or a subset of `label-vec` specified by `split`, if it is defined."
  [label-vec n-labels split]
  (if (some? split)
    (first (reduce (fn [[labels remaining-split] [row-id label]]
                     (if (pos? (get remaining-split label))
                       [(assoc labels row-id label)
                        (update remaining-split label dec)]
                       [labels remaining-split]))
                   [{} split]
                   label-vec))
    (into {} (take n-labels label-vec))))

(defn calc-results-argmax
  "Given the output from `search`, apply arg-max to each datum."
  [search-output]
  (reduce-kv (fn [m k v]
               (assoc m k (key (apply max-key val v))))
             {}
             search-output))

(defn calc-results-probs
  "Given the output from `search`, return the search probability of each datum."
  [search-output target-var]
  (reduce-kv (fn [output row-id probs]
               (assoc output row-id (get probs target-var)))
             {}
             search-output))

(defn bin-probabilities
  "Given a map of {row-id probs}, bins by row-id according to bin-size."
  [search-results n-bins]
  (let [bin-ticks (utils/linspace 0 1 n-bins)]
    (reduce (fn [bins [bin-min bin-max]]
              (conj bins (filter (comp #(and (< bin-min %)
                                             (<= % bin-max))
                                       #(Math/exp %)
                                       val)
                                 search-results)))
            []
            (partition 2 1 bin-ticks))))

(defn confusion-matrix
  "Given a map of {row-id predictions} and {row-id ground-truth}, computes
  the confusion matrix in the form of {actual {predicted count}}."
  [predictions ground-truth]
  (assert (= (count predictions) (count ground-truth)))
  (reduce (fn [results row-id]
            (let [predicted (get predictions row-id)
                  actual (get ground-truth row-id)]
              (update-in results [actual predicted] (fnil inc 0))))
          {}
          (-> ground-truth count (range))))

(defn false-positive
  "Given a confusion-matrix, calculates the false-positive rate for the variable cls."
  [confusion-matrix cls]
  (reduce-kv (fn [m _ stats]
               (+ m (get stats cls 0)))
             0
             (dissoc confusion-matrix cls)))

(defn false-negative
  "Given a confusion-matrix, calculates the false-negative rate for the variable cls."
  [confusion-matrix cls]
  (reduce + (-> confusion-matrix
                (get cls)
                (dissoc cls)
                (vals))))

(defn true-positive
  "Given a confusion-matrix, calculates the true-positive rate for the variable cls."
  [confusion-matrix cls]
  (get-in confusion-matrix [cls cls]))

(defn true-negative
  "Given a confusion-matrix, calculates the true-negative rate for the variable cls."
  [confusion-matrix cls]
  (reduce-kv (fn [m cls' stats]
               (+ m (reduce + (-> stats (dissoc cls') vals))))
             0
             (dissoc confusion-matrix cls)))

(defn precision-recall
  "Given a confusion-matrix and list of possible classes, computes
  for each class the precision, recall, and specificity."
  [confusion-matrix classes]
  (reduce (fn [pr-rec cls]
            (let [true-pos (true-positive confusion-matrix cls)
                  true-neg (true-negative confusion-matrix cls)
                  false-pos (false-positive confusion-matrix cls)
                  false-neg (false-negative confusion-matrix cls)
                  total-pos (+ true-pos false-pos)
                  precision (double (/ true-pos (if (pos? total-pos) total-pos 1)))
                  recall (double (/ true-pos (+ true-pos false-neg)))
                  total-neg (+ true-neg false-pos)
                  specificity (double (/ true-neg (if (pos? total-neg) total-neg 1)))]
              (assoc pr-rec cls {:precision precision
                                 :recall recall
                                 :specificity specificity})))
          {}
          classes))

(defn generate-vega-ultimate
  [confidence-accuracy-metrics n-labels x-axis y-axis]
  (json/write-str
   {"$schema" "https://vega.github.io/schema/vega-lite/v4.json"
    "data" {"values" (vals confidence-accuracy-metrics)}
    "title" (str "Confidence v. accuracy with " n-labels " labeled rows")
    "layer" [{"mark" {"type" "point"
                      "filled" true}
              "encoding" {"x" {"field" x-axis
                               "type" "quantitative"
                               "scale" {"domain" [0 1]}}
                          "y" {"field" y-axis
                               "type" "quantitative"}
                          "scale" {"domain" [0 1]}}}
             {"mark" {"type" "line"
                      "color" "firebrick"}
              "transform" [{"regression" y-axis
                            "on" x-axis
                            "extent" [0 1]}]
              "encoding" {"x" {"field" x-axis
                               "type" "quantitative"
                               "scale" {"domain" [0 1]}}
                          "y" {"field" y-axis
                               "type" "quantitative"
                               "scale" {"domain" [0 1]}}}}
             {"transform" [{"regression" y-axis
                            "on" x-axis
                            "params" true}
                           {"calculate" "'R²: '+format(datum.rSquared, '.2f')"
                            "as" "R2"}]
              "mark" {"type" "text"
                      "color" "firebrick"
                      "x" "width"
                      "align" "right"
                      "y" -5}
              "encoding" {"text" {"type" "nominal"
                                  "field" "R2"}}}]}))
