(ns inferenceql.inference.gpm.multimixture.search
  (:require [clojure.spec.alpha :as s]
            [metaprob.distributions :as dist]
            [metaprob.generative-functions :as g :refer [at gen]]
            [metaprob.prelude :as mp]
            [inferenceql.inference.distributions :as idbdist]
            [inferenceql.inference.gpm.multimixture.utils :as mmix-utils]
            [inferenceql.inference.gpm :as gpm]
            [inferenceql.inference.utils :as utils]
            [inferenceql.inference.gpm.multimixture.specification :as spec]))

(s/fdef optimized-row-generator
  :args (s/cat :spec ::spec/multi-mixture))

(defn optimized-row-generator
  [spec]
  (let [row-generator (mmix-utils/row-generator spec)]
    (g/make-generative-function
     row-generator
     (gen [partial-trace]
       (let [all-latents    (mmix-utils/all-latents spec)
             all-traces     (mapv #(merge partial-trace %)
                                  all-latents)
             all-logscores  (mapv #(last (mp/infer-and-score :procedure row-generator
                                                             :observation-trace %))
                                  all-traces)
             all-scores (map mp/exp all-logscores)
             all-zeroes (every? #(== 0 %) all-scores)
             log-normalizer (if all-zeroes ##-Inf (dist/logsumexp all-logscores))
             score          log-normalizer
             categorical-params (if all-zeroes
                                  (mmix-utils/uniform-categorical-params (count all-scores))
                                  (dist/normalize-numbers all-scores))]
         (gen []
           (let [i     (dist/categorical categorical-params)
                 trace (nth all-traces i)
                 v     (first (mp/infer-and-score :procedure row-generator
                                                  :observation-trace trace))]
             [v trace score])))))))

(s/fdef generate-1col-binary-extension
  :args (s/cat :spec ::spec/multi-mixture
               :row-count pos-int?
               :colunn-key ::spec/column
               :beta-parameters ::spec/beta-parameters))

(def generate-1col-binary-extension
  (gen [spec row-count column-key {:keys [alpha beta]}]
    (let [view-idx (at :view dist/categorical (mmix-utils/uniform-categorical-params (count (:views spec))))
          new-spec (-> spec
                       (assoc-in [:vars column-key] :binary)
                       (update-in [:views view-idx]
                                  (fn [clusters]
                                    (vec (map-indexed (fn [i cluster]
                                                        (update cluster :parameters
                                                                #(assoc % column-key
                                                                        (at `(:cluster-parameters ~i)
                                                                            #?(:clj idbdist/beta
                                                                               :cljs idbdist/beta)
                                                                            {:alpha alpha
                                                                             :beta beta}))))
                                                      clusters)))))
          new-row-generator (optimized-row-generator new-spec)]
      (doseq [i (range row-count)]
        (at `(:rows ~i) new-row-generator))
      new-spec)))

(defn importance-resampling
  [model {:keys [inputs observation-trace n-particles]
          :or {inputs [], observation-trace {}, n-particles 1}}]
  (let [particles (mp/replicate n-particles
                                #(mp/infer-and-score :procedure model
                                                     :inputs inputs
                                                     :observation-trace observation-trace))]
    (nth particles (dist/log-categorical (map last particles)))))

(s/fdef insert-column
  :args (s/cat :spec ::spec/multi-mixture
               :rows ::spec/rows
               :column-key ::spec/column
               :beta-params ::spec/beta-parameters))

(defn insert-column
  "Takes a multimixture specification, views, and a set of rows that have a value
  in the new column that is being added. Returns an updated multimixture
  specification."
  [spec rows column-key beta-params {:keys [n-particles] :or {n-particles 100}}]
  (first
   ;; TODO: Setting n-particles to 1 causes IOB errors
   (importance-resampling generate-1col-binary-extension
                          {:inputs [spec (count rows) column-key beta-params]
                           :observation-trace (mmix-utils/with-rows {} rows)
                           :n-particles n-particles})))

(defn score-rows
  [spec rows new-column-key]
  (let [new-column-view (spec/view-index-for-variable spec new-column-key)
        row-generator (optimized-row-generator spec)]
    (mapv (fn [row]
            (let [[_ trace _] (mp/infer-and-score :procedure row-generator
                                                  :observation-trace (mmix-utils/with-row-values {} row))
                  cluster-idx (get-in trace [:cluster-assignments-for-view new-column-view :value])]
              (get-in spec [:views new-column-view cluster-idx :parameters new-column-key])))
          rows)))

(defn transpose
  [coll]
  (apply map vector coll))

(defn constraints-for-scoring-p
  [target-col constraint-cols row]
  (->> (if (= (first constraint-cols) "ROW")
         (remove (comp #{target-col} key)
                 row)
         (select-keys row constraint-cols))
       (remove (comp nil? val))
       (into {})))

(defn score-row-probability
  [row-generator target-col constraint-cols row]
  (let [target (select-keys row [target-col])
        constraints (constraints-for-scoring-p target-col constraint-cols row)]
    (if (nil? (get target target-col))
      1
      (Math/exp (gpm/logpdf (gpm/Multimixture row-generator) target constraints)))))

(defn anomaly-search
  [spec target-col conditional-cols data]
  (let [row-generator (optimized-row-generator spec)]
    (map #(score-row-probability row-generator target-col conditional-cols %) data)))

(defn search
  "Conducts classic search via importance sampling, parallelized. If only one spec is given, there
  will be `n-models` models based on that spec. Otherwise, the number of specs
  in `spec` must equal n-models."
  ;; Additional arity for calling search without an options map.
  ([spec new-column-key known-rows unknown-rows n-models beta-params]
   (search spec new-column-key known-rows unknown-rows n-models beta-params {}))
  ([spec new-column-key known-rows unknown-rows n-models beta-params {:keys [n-particles] :or {n-particles 100}}]
   (let [pmap #?(:clj pmap
                 :cljs map)
         specs (if (seq spec)
                 (utils/prun n-models #(insert-column spec known-rows new-column-key beta-params {:n-particles n-particles}))
                 (pmap #(insert-column % known-rows new-column-key beta-params {:n-particles n-particles}) spec))
         predictions (mapv #(score-rows % unknown-rows new-column-key)
                           specs)]
     (into []
           (map #(/ (reduce + %)
                    n-models))
           (transpose predictions)))))
