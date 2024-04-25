(ns gensql.inference.gpm.multimixture.utils
  #?(:cljs (:require-macros [metaprob.generative-functions :refer [gen]]))
  (:require [clojure.math.combinatorics :as combo]
            [clojure.spec.alpha :as s]
            #?(:clj [metaprob.generative-functions :as g :refer [apply-at at gen]]
               :cljs [metaprob.generative-functions :as g :refer [apply-at at]])
            [metaprob.distributions :as dist]
            [metaprob.prelude :as mp]
            [gensql.inference.gpm.multimixture.specification :as spec]))

(defn row-generator
  "Returns a generative function that samples a row from the provided view
  specification."
  [{:keys [vars views]}]
  (gen []
    (let [partial-row (fn partial-row [i clusters]
                        (let [cluster-idx (at `(:cluster-assignments-for-view ~i)
                                              dist/categorical
                                              (mapv :probability clusters))
                              cluster (nth clusters cluster-idx)]
                          (reduce-kv (fn [m variable params]
                                       (let [primitive (case (get vars variable)
                                                         :binary dist/flip
                                                         :gaussian dist/gaussian
                                                         :categorical dist/categorical)
                                             params (case (get vars variable)
                                                      :binary [params]
                                                      :gaussian [(:mu params) (:sigma params)]
                                                      :categorical [params])]
                                         (assoc m variable (apply-at `(:columns ~variable) primitive params))))
                                     {}
                                     (:parameters cluster))))]
      (->> views
           (map-indexed partial-row)
           (reduce merge)))))

(defn cluster-row-generator
  "Returns a generative function that samples a partial row from the provided cluster
  in a given view, with respect to a given specification."
  [cluster vars]
  (gen []
    (reduce-kv
     (fn [m variable params]
       (let [primitive (case (get vars variable)
                         :binary dist/flip
                         :gaussian dist/gaussian
                         :categorical dist/categorical)
             params (case (get vars variable)
                      :binary [params]
                      :gaussian [(:mu params) (:sigma params)]
                      :categorical [params])]
         (assoc m variable (apply-at `(:columns ~variable) primitive params))))
     {}
     (:parameters cluster))))

(defn with-cluster-assignment
  "Sets the cluster assignment in trace for view index view-i to cluster index
  cluster-i."
  [trace view-i cluster-i]
  (assoc-in trace [:cluster-assignments-for-view view-i :value] cluster-i))

(defn with-cell-value
  "Sets the cell value in trace for variable var to value v."
  [trace var v]
  (assoc-in trace [:columns var :value] v))

(defn with-row-values
  "Sets the values in the trace for cells in row to their values."
  [trace row]
  (reduce-kv (fn [trace var v]
               (with-cell-value trace var v))
             trace
             row))

(defn uniform-categorical-params
  [n]
  (repeat n (double (/ 1 n))))

(defn with-rows
  "Given a trace for generate-1col, produces a trace with the values in rows
  constrained."
  [trace rows]
  (assoc trace :rows (reduce (fn [acc [i row]]
                               (assoc acc i (with-row-values {} row)))
                             {}
                             (map-indexed vector rows))))

(s/fdef all-latents
  :args (s/cat :spec ::spec/multi-mixture))

(defn all-latents
  "Returns a lazy sequence of all the possible traces of latents."
  [spec]
  (->> (:views spec)
       (map (comp range count))
       (apply combo/cartesian-product)
       (map (fn [assignments]
              {:cluster-assignments-for-view
               (->> assignments
                    (map-indexed (fn [view-i cluster-i]
                                   {view-i {:value cluster-i}}))
                    (into {}))}))))

(defn optimized-row-generator
  [spec]
  (let [row-generator (row-generator spec)]
    (g/make-generative-function
     row-generator
     (gen [partial-trace]
       (let [all-latents    (all-latents spec)
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
                                  (uniform-categorical-params (count all-scores))
                                  (dist/normalize-numbers all-scores))]
         (gen []
           (let [i     (dist/categorical categorical-params)
                 trace (nth all-traces i)
                 v     (first (mp/infer-and-score :procedure row-generator
                                                  :observation-trace trace))]
             [v trace score])))))))
