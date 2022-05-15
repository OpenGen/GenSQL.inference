(ns inferenceql.inference.primitives
  (:require [clojure.math :as math]
            [inferenceql.inference.utils :as utils]
            #?(:cljs [inferenceql.inference.distributions :as dist]))
  #?(:clj (:import [org.apache.commons.math3.special Gamma]
                   [org.apache.commons.math3.random RandomDataGenerator])))

(defn gammaln
  "Returns the log of `x` under a gamma function."
  [x]
  (if-not x
   0
   #?(:clj (Gamma/logGamma x)
      :cljs (dist/log-gamma x))))

(defn betaln
  "Computes the natural log of the beta function.
  Assumes both arguments are greater than zero."
  [x y]
  (+ (gammaln x)
     (gammaln y)
     (- (gammaln (+ x y)))))

(defn bernoulli-logpdf
  "Returns log probability of `x` under a bernoulli distribution parameterized
  by `p`."
  [x {:keys [:p]}]
  (if x
    (math/log p)
    (math/log (- 1 p))))

(defn bernoulli-simulate
  "Generates a sample from a bernoulli distribution with parameter `p`.
  Generates `n` samples, if specified."
  ([{:keys [p]}]
   (let [flip (rand)]
     (< flip p)))
  ([n p]
   (repeatedly n #(bernoulli-simulate p))))

(defn gaussian-logpdf
  "Returns log probability of `x` under a gaussian distribution parameterized
  by shape parameter `mu`, with optional scale parameter `sigma`."
  [x {:keys [mu sigma]}]
  (let [z-inv (* -0.5 (+ (math/log sigma)
                         (math/log 2.0)
                         (math/log math/PI)))
        px    (* -0.5 (math/pow (/ (- x mu)
                                   sigma)
                                2.0))]
    (+ z-inv px)))

(defn gaussian-simulate
  "Generates a sample from a dirichlet distribution with vector parameter `alpha`.
  Based on a Box-Muller transform.
  Generates `n` samples, if specified."
  ([{:keys [mu sigma]}]
   (let [u1 (rand)
         u2 (rand)
         z0 (* (math/sqrt (* -2 (math/log u1)))
               (math/cos (* 2 math/PI u2)))]
     (+ (* z0 sigma) mu)))
  ([n parameters]
   (repeatedly n #(gaussian-simulate parameters))))

(defn gamma-logpdf
  "Returns log probability of `x` under a gamma distribution parameterized
  by shape parameter `k`, with optional scale parameter `theta`."
  [x {:keys [k theta]}]
  (let [z-inv (- (+ (gammaln k)
                    (* k (math/log theta))))
        px (- (* (- k 1)
                 (math/log x))
              (/ x theta))]
    (+ z-inv px)))

(defn gamma-simulate-webppl
  "An implementation of Marsaglia & Tang, 2000: A Simple Method for Generating Gamma Variables.
  Based on the javascript implementation found in WebPPL.

  This function works in CLJ as well as CLJS for testing purposes--although it is intened that
  we only use it in CLJS in practice."
  [{:keys [k theta]}]
  (if (< k 1)
    (let [r (* (gamma-simulate-webppl {:k (+ 1 k) :theta theta})
               (math/pow (rand) (/ 1 k)))
          ret-val (if (= r 0)
                    #?(:clj (Double/MIN_VALUE)
                       :cljs (.-MIN_VALUE js/Number))
                    r)]

      ;; Emit warning on underflow.
      (when (= r 0)
        (let [msg (str "gamma-simulate underflow. Rounded result to nearest "
                       "representable support value")]
          #?(:clj (.println (System/err) msg)
             :cljs (.warn js/console msg))))

      ret-val)
    (let [d (- k (/ 1 3))
          c (/ 1 (math/sqrt (* 9 d)))]
      (loop []
        (let [[x v] (loop []
                      (let [x (gaussian-simulate {:mu 0 :sigma 1})
                            v (+ 1 (* c x))]
                        (if (<= v 0)
                          (recur)
                          [x v])))
              v (* v v v)
              u (rand)]
          (if (or (< u (- 1 (* 0.331 x x x x)))
                  (< (math/log u) (+ (* 0.5 x x)
                                     (* d (+ 1 (- v) (math/log v))))))
            (* theta d v)
            (recur)))))))

#?(:clj (def apache-sampler (RandomDataGenerator.)))
#?(:clj (defn gamma-simulate-apache
          [{:keys [k theta]}]
          (.nextGamma apache-sampler k theta)))

(defn gamma-simulate
  ([{:keys [k theta] :or {theta 1}}]
   #?(:cljs (gamma-simulate-webppl {:k k :theta theta})
      :clj (gamma-simulate-apache {:k k :theta theta})))
  ([n params]
   (repeatedly n #(gamma-simulate params))))

(defn beta-logpdf
  "Returns log probability of `x` under a beta distribution parameterized by
  `alpha` and `beta`."
  [x {:keys [alpha beta]}]
  (assert (and (pos? alpha) (pos? beta))
          (str "alpha and beta must be positive (" alpha ", " beta ")"))
  (let [k (- (gammaln (+ alpha beta))
             (+ (gammaln alpha)
                (gammaln beta)))
        c (- alpha 1)
        d (- beta 1)]
    (+ k (* c (math/log x))
       (* d (math/log (- 1 x))))))

(defn beta-simulate
  "Generates a sample from a beta distribution with parameters `alpha` and `beta`.
  Based on the method specified in Section I, of 'Generating Gamma and Beta Random Variables
  with Non-Integral Shape Parameters' by J Whittaker, at https://www.jstor.org/stable/pdf/2347003.pdf?seq=1 .
  Generates `n` samples, if specified."
  ([{:keys [alpha beta]}]
    (let [x1 (gamma-simulate {:k alpha})
          x2 (gamma-simulate {:k beta})]
      (/ x1 (+ x1 x2))))
  ([n parameters]
   (repeatedly n #(beta-simulate parameters))))

(defn categorical-logpdf
  "Log PDF for categorical distribution."
  [x {:keys [p]}]
  (let [prob (get p x)]
    (if (or (zero? prob) (nil? prob))
      ##-Inf
      (math/log prob))))

(defn categorical-simulate
  "Generates a sample from a categorical distribution with vector parameters `p`.
  Generates `n` samples, if specified."
  ([{:keys [p]}]
    (let [p-sorted (sort-by last p)
          cdf (second (reduce (fn [[total v] [variable p]]
                                 (let [new-p (if (= total 0)
                                               p
                                               (+ total p))
                                       new-entry [variable new-p]]
                                   [new-p (conj v new-entry)]))
                               [0 []]
                               p-sorted))
          flip (rand)]
      (ffirst (drop-while #(< (second %) flip) cdf))))
  ([n parameters]
   (repeatedly n #(categorical-simulate parameters))))

(defn log-categorical-logpdf
  "Log PDF for categorical distribution parameterized by log probabilities."
  [x {:keys [p]}]
  (let [prob (get p x)]
    (if (nil? prob)
      ##-Inf
      prob)))

(defn log-categorical-simulate
  "Generates a sample from a categorical distribution with parameters `ps`,
  which are log probabilities.
  Generates `n` samples, if specified."
  ([{:keys [p]}]
    (let [p-sorted (sort-by last p)
          cdf (second (reduce (fn [[total v] [variable p]]
                                (let [new-p (if (zero? total)
                                              p
                                              (+ (utils/logsumexp [total p])))
                                      new-entry [variable new-p]]
                                  [new-p (conj v new-entry)]))
                              [0 []]
                              p-sorted))
          flip (math/log (rand))]
      (ffirst (drop-while #(< (second %) flip) cdf))))
  ([n p]
   (repeatedly n #(log-categorical-simulate p))))

(defn crp-logpdf
  "Returns log probability of table counts `x` under a Chinese Restaurant Process
  parameterized by a number `alpha`."
  [x {:keys [alpha]}]
  (let [n (reduce + x)
        p-tilde (+ (* (count x) (math/log alpha))
                   (->> x
                        (map #(gammaln %))
                        (reduce +)))
        z (- (gammaln (+ n alpha))
             (gammaln alpha))]
    (- p-tilde z)))

(defn crp-simulate
  "Simulates n customers of a Chinese Restaurant Process parameterized by
  the concentration parameter `alpha`.
  Returns a vector of [table-counts customer-assignments]."
  ([{:keys [alpha]}]
   (crp-simulate 1 {:alpha alpha}))
  ([n {:keys [alpha]}]
   (reduce (fn [[counts assignments] _]
             (let [probs-tilde  (conj counts alpha)
                   z (reduce + probs-tilde)
                   probs (zipmap (range)
                                 (map #(/ % z) probs-tilde))
                   c-i (categorical-simulate {:p probs})
                   assignments' (conj assignments c-i)
                   counts' (update counts c-i (fnil inc 0))]
               [counts' assignments']))
           [[1] [0]]
           (range 1 n))))

(defn crp-simulate-counts
  "Simulates a table assignment from a CRP with the given concentration parameter
  `alpha`. `counts` must be a coll of integers."
  [{:keys [alpha counts]}]
  (let [probs-tilde (assoc counts (gensym) alpha)
        z (reduce + (vals probs-tilde))
        probs (reduce-kv (fn [p category cnt]
                          (assoc p category (/ cnt z)))
                         {}
                         probs-tilde)]
    (categorical-simulate {:p probs})))

(defn dirichlet-logpdf
  "Returns log probability of `x` under a dirichlet distribution parameterized by
  a vector `alpha`."
  [x {:keys [alpha]}]
  (assert (= (count alpha) (count x)) "alpha and x must have same length")
  (let [z-inv (- (->> alpha
                      (map gammaln)
                      (reduce +))
                 (gammaln (reduce + alpha)))
        logDirichlet (apply + (map (fn [alpha-k x-k]
                                     (* (- alpha-k 1)
                                        (math/log x-k)))
                                   alpha
                                   x))]
    (+ z-inv logDirichlet)))

(defn dirichlet-simulate
  "Generates a sample from a dirichlet distribution with vector parameter `alpha`.
  Generates `n` samples, if specified."
  ([{:keys [alpha]}]
    (let [y (map #(gamma-simulate {:k % :theta 1}) alpha)
          z (reduce + y)]
      (mapv #(/ % z) y)))
  ([n parameters]
   (repeatedly n #(dirichlet-simulate parameters))))

(defn logpdf
  "Given a primitive, its parameters, returns the log probability of
  `x` under said primitive."
  ([x primitive parameters]
   (case primitive
     :bernoulli (bernoulli-logpdf x parameters)
     :beta (beta-logpdf x parameters)
     :categorical (categorical-logpdf x parameters)
     :crp (crp-logpdf x parameters)
     :dirichlet (dirichlet-logpdf x parameters)
     :gamma (gamma-logpdf x parameters)
     :gaussian (gaussian-logpdf x parameters)
     :log-categorical (log-categorical-logpdf x parameters)
     (throw (ex-info (str  "Primitive doesn't exist: " primitive) {:primitive primitive}))))
  ([primitive parameters]
   (partial logpdf primitive parameters)))

(defn simulate
  "Given a primitive and its parameters, generates a sample from the primitive.
  Generates `n` samples, if specified."
  ([primitive parameters]
   (case primitive
     :bernoulli (bernoulli-simulate parameters)
     :beta (beta-simulate  parameters)
     :categorical (categorical-simulate parameters)
     :dirichlet (dirichlet-simulate parameters)
     :gamma (gamma-simulate parameters)
     :gaussian (gaussian-simulate parameters)
     :log-categorical (log-categorical-simulate parameters)
     (throw (ex-info (str  "Primitive doesn't exist: " primitive) {:primitive primitive}))))
  ([n primitive parameters]
   (repeatedly n #(simulate primitive parameters))))
