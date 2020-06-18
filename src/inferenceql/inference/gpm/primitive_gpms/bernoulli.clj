(ns inferenceql.inference.gpm.primitive-gpms.bernoulli
  (:require [inferenceql.inference.gpm.proto :as gpm.proto]
            [inferenceql.inference.utils :as utils]
            [inferenceql.inference.primitives :as primitives]))

(defrecord Bernoulli [var-name parameters suff-stats hyperparameters]
  gpm.proto/GPM
  (logpdf [this targets constraints]
    (let [x-sum (get suff-stats :x-sum)
          n (get suff-stats :n)
          alpha' (+ (:alpha hyperparameters) x-sum)
          beta' (+ (:beta  hyperparameters) n (* -1 x-sum))
          denom  (Math/log (+ alpha' beta'))
          x (get targets var-name)
          x' (get constraints var-name)
          constrained? (not (nil? x'))]
      (cond
        (nil? x) 0
        constrained? (if (= x x') 0 ##-Inf)
        x (- (Math/log alpha') denom)
        :else (- (Math/log beta') denom))))
  (simulate [this targets constraints n-samples]
    (repeatedly n-samples #(< (Math/log (rand))
                              (gpm.proto/logpdf this {var-name true} {}))))

  gpm.proto/Incorporate
  (incorporate [this values]
    (let [x (get values var-name)]
      (-> this
          (assoc :suff-stats (-> suff-stats
                                 (update :n inc)
                                 (update :x-sum #(+ % (if x 1 0))))))))
  (unincorporate [this values]
    (let [x (get values var-name)]
      (-> this
          (assoc :suff-stats (-> suff-stats
                                 (update :n dec)
                                 (update :x-sum #(- % (if x 1 0))))))))

  gpm.proto/Score
  (logpdf-score [this]
    (let [n (:n suff-stats)
          x-sum (:x-sum suff-stats)
          alpha (:alpha hyperparameters)
          beta (:beta hyperparameters)]
      (- (primitives/betaln (+ alpha x-sum)
                            (+ n (- x-sum) beta))
         (primitives/betaln alpha beta)))))

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
  "Casts a CrossCat category spec to a Bernoulli variable."
  ([var-name]
   (spec->bernoulli var-name {:p 0.5} {:n 0 :x-sum 0}))
  ([var-name parameters]
   (spec->bernoulli var-name parameters {:n 0 :x-sum 0}))
  ([var-name parameters suff-stats]
   (spec->bernoulli var-name parameters suff-stats {:alpha 0.5 :beta 0.5}))
  ([var-name parameters suff-stats hyperparameters]
   (->Bernoulli var-name parameters suff-stats hyperparameters)))
