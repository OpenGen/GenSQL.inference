(ns inferenceql.inference.gpm.primitive-gpms.categorical
  (:require [inferenceql.inference.gpm.proto :as gpm.proto]
            [inferenceql.inference.utils :as utils]
            [inferenceql.inference.primitives :as primitives]))

(defrecord Categorical [var-name suff-stats hyperparameters]
  gpm.proto/GPM
  (logpdf [this targets constraints]
    (let [x (get targets var-name)
          x' (get constraints var-name)
          constrained? (not (nil? x'))]
      (cond
        (nil? x) 0
        constrained? (if (= x x') 0 ##-Inf)
        :else (let [counts (:counts suff-stats)
                    alpha  (:alpha hyperparameters)
                    numer  (Math/log (+ alpha (get counts x)))
                    denom  (Math/log (+ (* alpha (count counts))
                                        (reduce + (vals counts))))]
                (- numer denom)))))
  (simulate [this targets constraints n-samples]
    (let [p (->> (keys (:counts suff-stats))
                 (reduce (fn [m k]
                           (assoc m k (gpm.proto/logpdf
                                       this
                                       {var-name k}
                                       {})))
                         {})
                 (assoc {} :p))]
    (primitives/simulate n-samples :log-categorical p)))

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
  (logpdf-score [this]
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
         (* -1 k (primitives/gammaln alpha))))))

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
