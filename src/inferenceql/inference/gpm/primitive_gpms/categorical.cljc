(ns inferenceql.inference.gpm.primitive-gpms.categorical
  (:require [clojure.math :as math]
            [inferenceql.inference.gpm.proto :as gpm.proto]
            [inferenceql.inference.primitives :as primitives]
            [inferenceql.inference.utils :as utils]))

(defrecord Categorical [var-name suff-stats hyperparameters]
  gpm.proto/GPM
  (logpdf [_ targets constraints]
    (let [x (get targets var-name)
          x' (get constraints var-name)
          constrained? (not (nil? x'))]
      (cond
        (nil? x) 0
        constrained? (if (= x x') 0 ##-Inf)
        :else (let [counts (:counts suff-stats)
                    alpha  (:alpha hyperparameters)
                    numer  (math/log (+ alpha (get counts x)))
                    denom  (math/log (+ (* alpha (count counts))
                                        (reduce + (vals counts))))]
                (- numer denom)))))
  (simulate [this _ _]
    (let [p (->> (keys (:counts suff-stats))
                 (reduce (fn [m k]
                           (assoc m k (gpm.proto/logpdf
                                       this
                                       {var-name k}
                                       {})))
                         {})
                 (assoc {} :p))]
      (primitives/simulate :log-categorical p)))

  gpm.proto/Incorporate
  (incorporate [this values]
    (let [x (get values var-name)]
      (-> this
          (assoc :suff-stats (-> suff-stats
                                 (update :n inc)
                                 (update-in [:counts x] (fnil inc 0)))))))
  (unincorporate [this values]
    (let [x (get values var-name)]
      (-> this
          (assoc :suff-stats (-> suff-stats
                                 (update :n dec)
                                 (update-in [:counts x] dec))))))

  gpm.proto/Score
  (logpdf-score [_]
    (let [counts (:counts suff-stats)
          n (:n suff-stats)
          k (count counts)
          alpha (:alpha hyperparameters)
          a (* k alpha)
          lg (reduce +
                     (map (fn [v] (primitives/gammaln (+ v alpha)))
                          (vals counts)))]
      (+ (primitives/gammaln a)
         (- (primitives/gammaln (+ a n)))
         lg
         (* -1 k (primitives/gammaln alpha)))))

  gpm.proto/Variables
  (variables [{:keys [var-name]}]
    #{var-name}))

(defn categorical?
  "Checks if the given pGPM is Categorical."
  [stattype]
  (and (record? stattype)
       (instance? Categorical stattype)))

(defn hyper-grid
  "Hyperparameter grid for the Categorical variable, used in column hyperparameter inference
  for Column GPMs."
  [data n-grid]
  (let [grid (utils/log-linspace 1 (count data) n-grid)]
    {:alpha grid}))

(defn spec->categorical
  "Casts a CrossCat category spec to a Categorical pGPM.
  Requires a variable name, optionally takes by key
  sufficient statistics, options, and hyperparameters."
  [var-name & {:keys [hyperparameters suff-stats options]}]
  (let [suff-stats' (if (and (nil? suff-stats) (not (nil? options)))
                      {:n 0 :counts (zipmap options (repeat 0))}
                      suff-stats)
        hyperparameters' (if-not (nil? hyperparameters) hyperparameters {:alpha 1})]
    (->Categorical var-name suff-stats' hyperparameters')))

(defn export
  "Exports a Categorical pGPM to a Multimixture spec."
  [categorical]
  (let [counts (-> categorical :suff-stats :counts)
        n (-> categorical :suff-stats :n)
        alpha (-> categorical :hyperparameters :alpha)
        z (+ n (* (count counts) alpha))]
    {(:var-name categorical) (reduce-kv (fn [p option cnt]
                                          (assoc p option (double (/ (+ cnt alpha)  z))))
                                        {}
                                        counts)}))
