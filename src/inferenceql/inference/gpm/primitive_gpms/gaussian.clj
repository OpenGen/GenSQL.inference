(ns inferenceql.inference.gpm.primitive-gpms.gaussian
  (:require [inferenceql.inference.gpm.proto :as gpm.proto]
            [inferenceql.inference.utils :as utils]
            [inferenceql.inference.primitives :as primitives]))

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
  (+ (* 0.5 (+ (* nu (- (Math/log 2)
                        (Math/log s)))
               (Math/log Math/PI)
               (Math/log 2)
               (- (Math/log r))))
     (primitives/gammaln (/ nu 2))))

;; The Gaussian pGPM is defined as a Normal-Inverse-Gamma distribution,
;; which allows us to perform inference on a gaussian variable for which
;; both mean and variance are unknown.
;; Follow http://www.stats.ox.ac.uk/~teh/research/notes/GaussianInverseGamma.pdf
;; for further information.
(defrecord Gaussian [var-name suff-stats hyperparameters]
  gpm.proto/GPM
  (logpdf   [this targets constraints]
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
          (+ (* -0.5 (+ (Math/log 2) (Math/log Math/PI)))
                        z''
                       (* -1 z'))))))
  (simulate [this targets constraints n-samples]
    (let [[m-n r-n s-n nu-n] (posterior-hypers (:n suff-stats)
                                               (:sum-x suff-stats)
                                               (:sum-x-sq suff-stats)
                                               hyperparameters)]
    (repeatedly n-samples #(let [rho (primitives/simulate :gamma {:k (/ nu-n 2) :theta (/ 2 s-n)})
                                 mu (primitives/simulate :gaussian {:mu m-n :sigma (/ 1 (Math/pow (* rho r-n) 0.5))})]
      (primitives/simulate :gaussian {:mu mu :sigma (/ 1 rho)})))))

  gpm.proto/Incorporate
  (incorporate [this values]
    (let [x (get values var-name)]
      (-> this
          (assoc :suff-stats (-> suff-stats
                                 (update :n inc)
                                 (update :sum-x #(+ % x))
                                 (update :sum-x-sq #(+ % (* x x))))))))
  (unincorporate [this values]
    (let [x (get values var-name)]
      (-> this
          (assoc :suff-stats (-> suff-stats
                                 (update :n dec)
                                 (update :sum-x #(- % x))
                                 (update :sum-x-sq #(- % (* x x))))))))

  gpm.proto/Score
  (logpdf-score [this]
    (let [n                (:n suff-stats)
          sum-x            (:sum-x suff-stats)
          sum-x-sq         (:sum-x-sq suff-stats)
          [_ r-n s-n nu-n] (posterior-hypers n
                                             sum-x
                                             sum-x-sq
                                             hyperparameters)
          z-n              (calc-z r-n s-n nu-n)
          z-0              (calc-z (:r hyperparameters) (:s hyperparameters) (:nu hyperparameters))]
      (+ (* -0.5 n (+ (Math/log 2) (Math/log Math/PI)))
         z-n
         (- z-0)))))

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
        sum-sq-dev (* (utils/variance data) (- n 2))]
    {:m (utils/linspace (apply min data) (inc (apply max data)) n-grid)
     :r (utils/log-linspace (/ 1 n) n n-grid)
     :s (utils/log-linspace (/ sum-sq-dev 100.) sum-sq-dev n-grid)
     :nu (utils/log-linspace 1 n n-grid)}))

(defn spec->gaussian
  "Casts a CrossCat category spec to a Gaussian pGPM.
  Requires a variable name, optionally takes by position
  sufficient statistics, and hyperparameters."
  ([var-name]
   (spec->gaussian var-name {:n 0 :sum-x 0 :sum-x-sq 0} {:m 0 :r 1 :s 1 :nu 1}))
  ([var-name suff-stats]
   (spec->gaussian var-name suff-stats {:m 0 :r 1 :s 1 :nu 1}))
  ([var-name suff-stats hyperparameters]
   (->Gaussian var-name suff-stats hyperparameters)))
