(ns inferenceql.inference.gpm.primitive-gpms.gaussian
  (:require
            [clojure.math :as math]
            [inferenceql.inference.event :as event]
            [inferenceql.inference.gpm.proto :as gpm.proto]
            [inferenceql.inference.primitives :as primitives]
            [inferenceql.inference.event :as event]
            [inferenceql.inference.utils :as utils])
  #?(:clj (:import [org.apache.commons.math3.distribution TDistribution])))

(defn posterior-hypers
  "Given sufficient statistics and the current hyperparameters,
  returns the hyperparameters of the posterior distribution."
  [n sum-x sum-x-sq hyperparameters]
  (let [m    (:m hyperparameters)
        r    (:r hyperparameters)
        s    (:s hyperparameters)
        nu   (:nu hyperparameters)
        r'  (+ r n)
        nu' (+ nu n)
        m'  (/ (+ (* r m)
                  sum-x)
               r')
        s'  (+ s
               sum-x-sq
               (* r m m)
               (* -1 r' m' m'))]
    [m' r' (if (zero? s') s s') nu']))

(defn calc-z
  "Given the hyperparameters r, s, and nu, calculates the normalizing
  constant Z of a Normal-Inverse-Gamma distribution."
  [r s nu]
  (+ (* 0.5 (+ (* nu (- (math/log 2.0)
                        (math/log s)))
               (math/log math/PI)
               (math/log 2.0)
               (- (math/log r))))
     (primitives/gammaln (/ nu 2))))

#?(:clj (defn student-t-cdf
  [value df loc scale]
  (let [scaled-and-shifted (/ (- value loc) scale)]
      (.cumulativeProbability (TDistribution. df) scaled-and-shifted))))

;; The Gaussian pGPM is defined as a Normal-Inverse-Gamma distribution,
;; which allows us to perform inference on a gaussian variable for which
;; both mean and variance are unknown.
;; Follow http://www.stats.ox.ac.uk/~teh/research/notes/GaussianInverseGamma.pdf
;; for further information.

(defrecord Gaussian [var-name suff-stats hyperparameters]
  gpm.proto/GPM
  (logpdf   [_ targets constraints]
    (let [x (get targets var-name)
          x' (get constraints var-name)
          constrained? (not (nil? x'))]
      (cond
        (nil? x) 0
        constrained? (if (= x x') 0 ##-Inf)
        :else (let [n (:n suff-stats)
                    sum-x (:sum-x suff-stats)
                    sum-x-sq (:sum-x-sq suff-stats)
                    [_ r' s' nu'] (posterior-hypers n
                                                    sum-x
                                                    sum-x-sq
                                                    hyperparameters)
                    ;; The below could be rewritten by augmenting the above values
                    ;; which would not be quite D.R.Y but would result in fewer
                    ;; mathematical operations.
                    [_ r'' s'' nu''] (posterior-hypers (inc n)
                                                       (+ sum-x x)
                                                       (+ sum-x-sq (* x x))
                                                       hyperparameters)
                    z' (calc-z r' s' nu')
                    z'' (calc-z r'' s'' nu'')]
                (+ (* -0.5 (+ (math/log 2.0) (math/log math/PI)))
                   z''
                   (* -1 z'))))))
  (simulate [_ _ _]
    (let [[m-n r-n s-n nu-n] (posterior-hypers (:n suff-stats)
                                               (:sum-x suff-stats)
                                               (:sum-x-sq suff-stats)
                                               hyperparameters)
          rho (primitives/simulate :gamma {:k (/ nu-n 2) :theta (/ 2 s-n)})
          mu (primitives/simulate :gaussian {:mu m-n :sigma (/ 1 (math/pow (* rho r-n) 0.5))})]
      (primitives/simulate :gaussian {:mu mu :sigma (math/pow rho -0.5)})))


  gpm.proto/LogProb
  #?(:clj (logprob [this event]
    (if (event/negated? event)
      (utils/log-diff 0 (gpm.proto/logprob this (second event)))
      (let [[operator a b] event
            {:keys [m r s nu]} hyperparameters
            {:keys [n sum-x sum-x-sq]} suff-stats
            value (cond (and (event/variable? a) (number? b)) b
                        (and (number? a) (event/variable? b)) a
                        :else (throw (ex-info "Cannot compute the log probability of an event without a variable."
                                              {:event event})))
            ;; Following https://github.com/probcomp/cgpm/blob/master/tests/test_teh_murphy.py
            ;; for the conversion of the hyperparameters into the parameters for a Student t
            ;; distribution (see also: https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf)
            rn  (+ r n)
            nun (+ nu n)
            mn  (/ (+ (* r m) sum-x) rn)
            sn  (+ s sum-x-sq (* r m m) (* -1 rn mn mn))
            an  (/ nun 2.0)
            bn  (/ sn 2.0)
            scalesq (/ (* bn (+ rn 1.0)) (* an rn))
            df (* 2 an)
            loc mn
            scale (math/sqrt scalesq)]
        (condp = operator
          '< (math/log (student-t-cdf value df loc scale))
          '> (math/log (- 1 (student-t-cdf value df loc scale)))
          (throw (ex-info "Computing the log probability for operators other than <, > is not yet supported."
                          {:operator operator})))))))

  gpm.proto/Incorporate
  (incorporate [this values]
    (let [x (get values var-name)]
      (assoc this :suff-stats (-> suff-stats
                                  (update :n inc)
                                  (update :sum-x #(+ % x))
                                  (update :sum-x-sq #(+ % (* x x)))))))
  (unincorporate [this values]
    (let [x (get values var-name)]
      (assoc this :suff-stats (-> suff-stats
                                  (update :n dec)
                                  (update :sum-x #(- % x))
                                  (update :sum-x-sq #(- % (* x x)))))))

  gpm.proto/Score
  (logpdf-score [_]
    (let [n                (:n suff-stats)
          sum-x            (:sum-x suff-stats)
          sum-x-sq         (:sum-x-sq suff-stats)
          [_ r-n s-n nu-n] (posterior-hypers n
                                             sum-x
                                             sum-x-sq
                                             hyperparameters)
          z-n              (calc-z r-n s-n nu-n)
          z-0              (calc-z (:r hyperparameters) (:s hyperparameters) (:nu hyperparameters))]
      (+ (* -0.5 n (+ (math/log 2.0) (math/log math/PI)))
         z-n
         (- z-0))))

  gpm.proto/Variables
  (variables [{:keys [var-name]}]
    #{var-name}))

(defn gaussian?
  "Checks if the given pGPM is Gaussian."
  [stattype]
  (and (record? stattype)
       (instance? Gaussian stattype)))

(defn hyper-grid
  "Hyperparameter grid for the Gaussian variable, used in column hyperparameter inference
  for Column GPMs.
  This mirrors the implementation in `cgpm` library for a `normal` variable."
  [data n-grid]
  (let [n (inc (count data))
        sum-sq-dev (* (utils/variance data) (- n 2))
        ;; Must ensure the sum of squares deviation is nonzero.
        sum-sq-dev (if (zero? sum-sq-dev) 0.01 sum-sq-dev)]
    {:m (utils/linspace (apply min data) (inc (+ (apply max data) 5)) n-grid)
     :r (utils/log-linspace (/ 1 n) n n-grid)
     :s (utils/log-linspace (/ sum-sq-dev 100.) sum-sq-dev n-grid)
     :nu (utils/log-linspace 1 n n-grid)}))

(defn spec->gaussian
  "Casts a CrossCat category spec to a Gaussian pGPM.
  Requires a variable name, optionally takes by key
  sufficient statistics and hyperparameters."
  [var-name & {:keys [hyperparameters suff-stats]}]
  (let [suff-stats' (if-not (nil? suff-stats) suff-stats {:n 0 :sum-x 0 :sum-x-sq 0})
        hyperparameters' (if-not (nil? hyperparameters) hyperparameters {:m 0 :r 1 :s 1 :nu 1})]
    (->Gaussian var-name suff-stats' hyperparameters')))

(defn export
  "Exports a Gaussian pGPM to a Multimixture spec."
  [gaussian]
  (let [samples (repeatedly 1000 #(gpm.proto/simulate gaussian [(:var-name gaussian)] {}))]
    {(:var-name gaussian) {:mu (utils/average samples) :sigma (utils/std samples)}}))
