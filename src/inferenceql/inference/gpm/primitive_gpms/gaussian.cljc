(ns inferenceql.inference.gpm.primitive-gpms.gaussian
  (:require [clojure.math :as math]
            [inferenceql.inference.event :as event]
            [inferenceql.inference.gpm.proto :as gpm.proto]
            [fastmath.random :as r]
            [inferenceql.inference.primitives :as primitives]
            [inferenceql.inference.utils :as utils]))

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

;; The Gaussian pGPM is defined as a Normal-Inverse-Gamma distribution,
;; which allows us to perform inference on a gaussian variable for which
;; both mean and variance are unknown.
;; Follow http://www.stats.ox.ac.uk/~teh/research/notes/GaussianInverseGamma.pdf
;; for further information.

;; Ulli: This should of course not be a constant
(def DEGREES-OF-FREEDOM 3)

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


  ;;(distribution :beta {:alpha 1.0, :beta 1.0})
  ;;>  :t (:degrees-of-freedom),
  gpm.proto/LogProb
  (logprob [_ event]
    (prn event)
    (let [[operator a b] event
          {:keys [m r s nu]} hyperparameters
          {:keys [n sum-x sum-x-sq]} suff-stats
          value (cond (and (symbol? a) (number? b)) b
                      (and (number? b) (symbol? a)) a
                      :else (throw (ex-info "Strange event" {:event event})))
          rn  (+ r n)
          nun (+ nu n)
          mn  (/(+ (* r m) sum-x) rn)
          sn  (+ s sum-x-sq (* r m m) (* -1 rn mn mn))
          an  (/ nun 2)
          bn  (/ sn 2)
          scalesq (/ (* bn (+ rn 1)) (* an rn))
          params {:degrees-of-freedom an :loc mn :scale (math/sqrt scalesq)}]
      (condp = operator
        '< (math/log (r/cdf (r/distribution :t params)
                            value))
        '> (math/log (- 1 (r/cdf (r/distribution :t params)
                                 value)))
        (throw (Exception. "Only simple events with < allowed for now")))))

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
    #{var-name})

  gpm.proto/MutualInfo
  (mutual-info [this event-a event-b]
    (let [operator first]
      (when-not (and (event/simple? event-a)
                     (event/simple? event-b))
        (throw (ex-info (str "Only simple events allowed! Operator was: " operator) {}))))
    (let [lpa1 (gpm.proto/logprob this event-a)
          lpb1 (gpm.proto/logprob this event-b)
          pa1 (math/exp lpa1)
          pb1 (math/exp lpb1)
          pa0 (- 1 pa1)
          pb0 (- 1 pb1)
          lpab1 (gpm.proto/logprob this `(~'and ~event-a ~event-b))]
      (for []))))

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
