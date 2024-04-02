(ns inferenceql.inference.gpm.primitive-gpms.poisson
  (:require [clojure.math :as math]
            [inferenceql.inference.gpm.proto :as gpm.proto]
            [inferenceql.inference.primitives :as primitives]
            [inferenceql.inference.distributions :as dist]
            [inferenceql.inference.utils :as utils]))

(defn posterior-hypers
 [n sum-x  a  b]
  [(+ a sum-x) (+ b n)])

(defn calc-log-Z
  [a b]
  (- (dist/log-gamma a) (* a (math/log b))))

(defrecord Poisson [var-name suff-stats hyperparameters]
  gpm.proto/GPM
  (logpdf [_ targets constraints]
    (let [x (get targets var-name)
          x' (get constraints var-name)
          constrained? (not (nil? x'))]
      (cond
        (nil? x) 0
        constrained? (if (= x x') 0 ##-Inf)
        :else (let [n (:n suff-stats)
                    sum-x  (:sum-x suff-stats)
                    sum-log-fact (:sum-log-fact suff-stats)
                    a  (:a hyperparameters)
                    b  (:b hyperparameters)
                    [an bn] (posterior-hypers n  sum-x  a  b)
                    [am bm] (posterior-hypers (+ n 1)  (+ sum-x x)  a  b)
                    Zn (calc-log-Z an bn)
                    Zm (calc-log-Z am bm)]
          (- Zm Zn (dist/log-gamma (+ x 1)))))))

  (simulate [this _ _]
    (throw (Exception. "Poisson simulate not implemented")))

  gpm.proto/Incorporate
  (incorporate [this values]
    (let [x (get values var-name)]
      (assoc this :suff-stats (-> suff-stats
                                  (update :n inc)
                                  (update :sum-x #(+ % x))
                                  (update :sum-log-fact #(+ % (log-gamma (+ x 1))))))))
  (unincorporate [this values]
    (let [x (get values var-name)]
      (assoc this :suff-stats (-> suff-stats
                                  (update :n dec)
                                  (update :sum-x #(- % x))
                                  (update :sum-log-fact #(- % (log-gamma (+ x 1))))))))

  gpm.proto/Variables
  (variables [{:keys [var-name]}]
    #{var-name}))

(defn poisson?
  "Checks if the given pGPM is Poisson."
  [stattype]
  (and (record? stattype)
       (instance? Poisson stattype)))

(defn hyper-grid
  "Hyperparameter grid for the Poisson variable, used in column hyperparameter inference
  for Column GPMs."
  [data n-grid]
  (let [grid (utils/log-linspace 1 (count data) n-grid)]
    {:alpha grid}))

(defn spec->poisson
  "Casts a CrossCat category spec to a Poisson pGPM.
  Requires a variable name, optionally takes by key
  sufficient statistics, options, and hyperparameters."
  [var-name & {:keys [hyperparameters suff-stats options]}]
  (let [suff-stats' (if-not (nil? suff-stats) suff-stats {:n 0 :sum-x 0 :sum-log-fact 0})
        hyperparameters' (if-not (nil? hyperparameters) hyperparameters {:a 1 :b 1})]
    (->Poisson var-name suff-stats' hyperparameters')))
