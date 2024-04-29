(ns gensql.inference.distributions
  (:require [clojure.math :as math]
            [clojure.spec.alpha :as s]
            [gensql.inference.gpm.multimixture.specification :as spec]
            [metaprob.distributions]
            [metaprob.prelude :as mp]))

(defn gamma
  "Returns Gamma(z + 1 = number) using Lanczos approximation.
  https://rosettacode.org/wiki/Gamma_function#Clojure"
  [number]
  (if (< number 0.5)
    (/ math/PI (* (math/sin (* math/PI number))
                  (gamma (- 1 number))))
    (let [n (dec number)
          c [0.99999999999980993 676.5203681218851 -1259.1392167224028
             771.32342877765313 -176.61502916214059 12.507343278686905
             -0.13857109526572012 9.9843695780195716e-6 1.5056327351493116e-7]]
      (* (math/sqrt (* 2 math/PI))
         (math/pow (+ n 7 0.5) (+ n 0.5))
         (math/exp (- (+ n 7 0.5)))
         (+ (first c)
            (apply + (map-indexed #(/ %2 (+ n %1 1)) (next c))))))))

(s/fdef gamma :args (s/cat :number pos?))

(defn log-gamma
  "Returns ln(Gamma(z + 1 = number)) using Lanczos approximation."
  [number]
  (if (< number 0.5)
    (- (math/log math/PI)
       (+ (math/log (math/sin (* math/PI number)))
          (log-gamma (- 1 number))))
    (let [n (dec number)
          c [0.99999999999980993 676.5203681218851 -1259.1392167224028
             771.32342877765313 -176.61502916214059 12.507343278686905
             -0.13857109526572012 9.9843695780195716e-6 1.5056327351493116e-7]]
      (+ (* 0.5 (math/log (* 2 math/PI)))
         (* (math/log (+ n 7 0.5))
            (+ n 0.5))
         (- (+ n 7 0.5))
         (math/log (+ (first c)
                      (apply + (map-indexed #(/ %2 (+ n %1 1)) (next c)))))))))

(defn gamma-dist [k]
  ;; Gamma usually takes a `s` also, but Ulrich says it's not needed.
  (loop []
    (let [e math/E
          u (rand)
          v (rand)
          w (rand)
          condition-1 (<= u (/ e (+ e k)))
          eps (if condition-1
                (math/pow v (/ 1 k))
                (- 1 (math/log v)))
          n   (if condition-1
                (* w (math/pow eps (- k 1)))
                (* w (math/pow e (- 0 eps))))
          condition-2 (> n (* (math/pow eps (- k 1))
                              (math/pow e (- 0 eps))))]
      (if condition-2 (recur) eps))))

(s/fdef beta-pdf
  :args (s/cat :v ::spec/probability
               :args (s/tuple ::spec/alpha ::spec/beta)))

(defn beta-pdf
  [v [alpha beta]]
  (let [b (fn [alpha beta]
            (/ (* (gamma alpha)
                  (gamma beta))
               (gamma (+ alpha beta))))]
    (/ (* (mp/expt v
                   (- alpha 1))
          (mp/expt (- 1 v)
                   (- beta 1)))
       (b alpha beta))))

(defn beta-logpdf
  [v [alpha beta]]
  (let [log-b (fn [alpha beta]
                (- (+ (log-gamma alpha)
                      (log-gamma beta))
                   (log-gamma (+ alpha beta))))]
    (- (+ (* (math/log v)
             (- alpha 1))
          (* (math/log (- 1 v))
             (- beta 1)))
       (log-b alpha beta))))


(s/fdef beta-sampler
  :args (s/cat :params ::spec/beta-parameters))

(defn- beta-sampler
  [{:keys [alpha beta]}]
  (let [x (gamma-dist alpha)
        y (gamma-dist beta)
        denom (+ x y)]
    (if (zero? denom)
      0
      (/ x denom))))

(s/fdef beta-scorer
  :args (s/cat :v ::spec/probability
               :args (s/tuple ::spec/beta-parameters)))

(defn- beta-scorer
  [v [{:keys [alpha beta]}]]
  (beta-logpdf v [alpha beta]))

(def beta (mp/make-primitive beta-sampler beta-scorer))
