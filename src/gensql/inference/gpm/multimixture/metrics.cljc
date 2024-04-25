(ns gensql.inference.gpm.multimixture.metrics
  (:require [clojure.math :as math]
            [gensql.inference.gpm.multimixture.utils :as utils]))

(defn check-distribution-criteria
  "Checks basic assumptions about two distributions being compared.
  Namely, the alphabets of the two distributions are the same, all
  entries are nonnegative, and the distributions sum to 1."
  ([p]
   (assert (< (abs (- (reduce + p) 1.0)) 1e-6) (str "Distribution doesn't sum to 1: "
                                                    (reduce + p)))
   (assert (every? #(>= % 0) p) "distribution contains negative elements"))
  ([p q]
   (assert (== (count p) (count q)) (str "p and q have different alphabet lengths: "
                                         (count p) " " (count q)))
   (check-distribution-criteria p)
   (check-distribution-criteria q)))

(defn kl-divergence
  "Calculates the Kullback-Leibler divergence between two distributions.
  `p` and `q` must have the same alphabet size (i.e. length).
  `p` must be zero when `q` is zero based on formal definition."
  [p q]
  (check-distribution-criteria p q)
  (let [result (reduce + (map (fn [pi qi]
                                (if (== qi 0)
                                  (if (== pi 0)
                                    0
                                    #?(:clj Integer/MIN_VALUE
                                       :cljs js/Number.MIN_SAFE_INTEGER))
                                  (* pi (math/log (/ pi qi)))))
                              p q))]
    ;; If the result is negative, that means pi != 0 when qi == 0 for all i
    (when-not (neg? result) result)))

(defn tv-distance
  "Calculates the total variational distance between two distributions.
  `p` and `q` must have the same alphabet size (i.e. length)."
  [p q]
  (check-distribution-criteria p q)
  (* 0.5 (reduce + (map (fn [pi qi] (abs (- pi qi)))
                        p q))))

(defn jensen-shannon-divergence
  "Calculates the Jenson-Shannon divergence between two distributions.
  Since it depends on
  `gensql.inference.multimixture.metrics/kl-divergence`, `p` must be zero
  when `q` is zero. `p` and `q` must have the same alphabet size (i.e. length)."
  [p q]
  (check-distribution-criteria p q)
  (let [m (map (fn [pi qi] (* 0.5 (+ pi qi))) p q)]
    (+ (* 0.5 (kl-divergence p m)) (* 0.5 (kl-divergence q m)))))

(defn generate-categorical-spec
  "Generates a spec to be used by `utils/row-generator` for a categorical variable of name
  `var-name` and probabilities `ps`."
  [var-name ps]
  (check-distribution-criteria ps)
  {:vars {var-name :categorical}
   :views [[{:probability 1
             :parameters {var-name (zipmap (map #(str %) (range (count ps))) ps)}}]]})

(defn generate-categorical-data
  "Generates `n` samples from a categorical distribution specified probabilities `ps`.
  If `ps` is a single float between 0 and 1, then it is assumed it's a binary distribution."
  [ps n & [var-name]]
  ;; Accounts for binary case.
  (let [var (or var-name "x")
        ps (if (vector? ps) ps [ps (- 1 ps)])
        spec (generate-categorical-spec var ps)
        generator (utils/row-generator spec)
        data (repeatedly n generator)]
    (doall (map #(get % var) data))))

(defn get-empirical-distribution
  "Calculates the empirical distribution of an array of datapoints. Each point must have a numerical class.
  `d` specifies dimensions of the data, defaults to 2 for the binary case."
  [data & [d]]
  (let [dims (or d 2)
        freqs (frequencies data)
        counts (map #(get freqs % 0) (map str (range dims)))
        n (count data)]
    (map #(/ % n) counts)))
