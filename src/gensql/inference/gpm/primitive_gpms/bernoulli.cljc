(ns gensql.inference.gpm.primitive-gpms.bernoulli
  (:require [clojure.math :as math]
            [gensql.inference.event :as event]
            [gensql.inference.gpm.proto :as gpm.proto]
            [gensql.inference.primitives :as primitives]
            [gensql.inference.utils :as utils]))

(defrecord Bernoulli [var-name suff-stats hyperparameters]
  gpm.proto/GPM
  (logpdf [_ targets constraints]
    (let [x-sum (get suff-stats :x-sum)
          n (get suff-stats :n)
          alpha' (+ (:alpha hyperparameters) x-sum)
          beta' (+ (:beta  hyperparameters) n (* -1 x-sum))
          denom  (math/log (+ alpha' beta'))
          x (get targets var-name)
          x' (get constraints var-name)
          constrained? (not (nil? x'))]
      (cond
        (nil? x) 0
        constrained? (if (= x x') 0 ##-Inf)
        x (- (math/log alpha') denom)
        :else (- (math/log beta') denom))))
  (simulate [this _ _]
    (< (math/log (rand))
       (gpm.proto/logpdf this {var-name true} {})))

  gpm.proto/Incorporate
  (incorporate [this values]
    (let [x (get values var-name)]
      (assert (boolean? x)
              "Only boolean values can be incorporated into a Bernoulli gpm.")
      (-> this
          (assoc :suff-stats (-> suff-stats
                                 (update :n inc)
                                 (update :x-sum #(+ % (if x 1 0))))))))
  (unincorporate [this values]
    (let [x (get values var-name)]
      (assert (boolean? x)
              "Only boolean values can be incorporated into a Bernoulli gpm.")
      (-> this
          (assoc :suff-stats (-> suff-stats
                                 (update :n dec)
                                 (update :x-sum #(- % (if x 1 0))))))))

  gpm.proto/Score
  (logpdf-score [_]
    (let [n (:n suff-stats)
          x-sum (:x-sum suff-stats)
          alpha (:alpha hyperparameters)
          beta (:beta hyperparameters)]
      (- (primitives/betaln (+ alpha x-sum)
                            (+ n (- x-sum) beta))
         (primitives/betaln alpha beta))))

  gpm.proto/LogProb
  (logprob [this event]
    (if (event/negated? event)
      (utils/log-diff 0 (gpm.proto/logprob this (second event)))
      (if (event/equality? event)
        (gpm.proto/logpdf this (event/eq-event-map event) {})
        (throw (ex-info "Strange event" {:event event})))))

  gpm.proto/Variables
  (variables [{:keys [var-name]}]
    #{var-name}))

(defn bernoulli?
  "Checks if the given pGPM is Bernoulli."
  [stattype]
  (and (record? stattype)
       (instance? Bernoulli stattype)))

(defn hyper-grid
  "Hyperparameter grid for the Bernoulli variable, used in column hyperparameter inference
  for Column GPMs."
  [data n-grid]
  (let [grid #(utils/log-linspace 0.5 (count data) n-grid)]
    {:alpha (grid) :beta (grid)}))

(defn spec->bernoulli
  "Casts a CrossCat category spec to a Bernoulli pGPM.
  Requires a variable name, optionally takes by key
  sufficient statistics and hyperparameters."
  [var-name & {:keys [hyperparameters suff-stats]}]
  (let [suff-stats' (if-not (nil? suff-stats) suff-stats {:n 0 :x-sum 0})
        hyperparameters' (if-not (nil? hyperparameters) hyperparameters {:alpha 0.5 :beta 0.5})]
    (->Bernoulli var-name suff-stats' hyperparameters')))

(defn export
  "Exports a Bernoulli pGPM to a Multimixture spec."
  [bernoulli]
  (let [x-sum (-> bernoulli :suff-stats :x-sum)
        n (-> bernoulli :suff-stats :n)
        alpha' (+ (-> bernoulli :hyperparameters :alpha) x-sum)
        beta' (+ (-> bernoulli :hyperparameters :beta) n (* -1 x-sum))]
    {(:var-name bernoulli) {:p (double (/ alpha' (+ alpha' beta')))}}))
