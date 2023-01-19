(ns inferenceql.inference.utils
  #?(:clj (:require [clojure.java.io :as io]))
  (:require [clojure.math :as math]))

(defn normalize [weights]
  (let [total (apply + weights)]
    (map #(/ % total) weights)))

(defn all? [l]
  (every? identity l))

(defn relerr [a b]
  (abs (- a b)))

(defn col
  [col-key table]
  (map (fn [row] (get row col-key)) table))

(defn average [column]
  (/ (reduce + column) (count column)))

(defn square [x] (* x x))

(defn variance [a]
  (if (= 1 (count a))
         0
    (/ (->> a
         (average)
         (repeat)
         (map - a)
         (map square)
         (reduce +))
       (- (count a) 1))))

(defn std [a]
  (math/sqrt (variance a)))

#?(:clj (defn save-json
          "Writes the provided Vega-Lite JSON to a file in the plots directory with the
          provided prefix."
          [file-prefix vl-json]
          (let [file-path (str "out/" file-prefix ".vl.json")]
            (io/make-parents file-path)
            (spit file-path vl-json))))

(defn column-subset [data columns]
  (let [row-subset (fn [row] (select-keys row columns))]
    (map row-subset data)))

(defn almost-equal?
  "Returns true if scalars `a` and `b` are approximately equal. Takes a distance
  metric (presumably from `inferenceql.metrics`) as its second argument. An example
  for a distance metric is Euclidean distance."
  [a b difference-metric threshold]
  (< (difference-metric a b) threshold))

(defn almost-equal-vectors?
  "Returns true if vectors `a` and `b` are approximately equal. Takes a difference
  metric (presumably from `inferenceql.metrics`) as its second argument."
  [a b difference-metric threshold]
  (assert (count a) (count b))
  (let [call-almost-equal
        (fn [i] (almost-equal?  (nth a i) (nth b i) difference-metric threshold))]
    (all? (map call-almost-equal (range (count a))))))

(defn almost-equal-maps?
  "Returns true if maps `a` and `b` are approximately equal. Takes a distance
  metric (presumably from `inferenceql.metrics`) as its second argument."
  [a b distance-metric threshold]
  (let [ks (keys a)]
    (almost-equal-vectors? (map #(get a %) ks)
                           (map #(get b %) ks)
                           distance-metric
                           threshold)))

(defn within-factor? [a b factor]
  (<= (/ b factor) a (* b factor)))

(defn probability-for-observed-categories [sample-vector]
  (let [fraction (fn [item] {(first (vals (first item)))
                             (float (/ (second item)
                                       (count sample-vector)))})
        occurences (frequencies sample-vector)]
    (apply merge (mapv fraction occurences))))

(defn probability-for-categories [sample-vector categories]
    (let [observed (probability-for-observed-categories sample-vector)]
      (into observed (keep (fn [category]
                             (when-not (contains? observed category)
                               [category 0.]))
                    categories))))

(defn probability-vector [samples possible-values]
  (let [probability-map (probability-for-observed-categories samples)]
    (map #(get probability-map % 0)
         possible-values)))


(defn equal-sample-values [samples-1 samples-2]
  (= (map (comp set vals) samples-1)
     (map (comp set vals) samples-2)))

(defn max-index
  "Returns the index of the maximum value in the provided vector."
  [xs]
  (first (apply max-key second (map-indexed vector xs))))

(defn pos-float? [value]
  (and (pos? value) (float? value)))

(defn prun
  "Runs `n` parallel calls to function `f`, that is assumed to have
  no arguments."
  [n f]
  #?(:clj (apply pcalls (repeat n f))
     :cljs (repeatedly n f)))

(defn transpose
  "Applies the standard tranpose operation to a collection. Assumes that
  `coll` is an object capable of having a transpose."
  [coll]
  (apply map vector coll))

(defn logsumexp
  "Log-sum-exp operation for summing log probabilities without
  leaving the log domain."
  [log-ps]
  (if (= 1 (count log-ps))
    (first log-ps)
    (let [log-ps-sorted (sort > log-ps)
          a0 (first log-ps-sorted)
          tail (drop 1 log-ps-sorted)
          res (+ a0 (math/log
                      (inc (reduce + (map #(math/exp (- % a0))
                                          tail)))))]
      #?(:clj (if (Double/isNaN res) ; A zero-probability event has occurred.
                  ##-Inf
                  res)
         :cljs (if (js/isNaN res)
                 ##-Inf
                 res)))))

(defn linspace
  "Generates a sequence of `n` numbers, linearly (evenly) spaced between `start` and `end`,
  the latter of which is exclusive."
  [start end n]
  (let [interval (/ (- end start) n)]
    (range start (+ end interval) interval)))

(defn log-linspace
  "Generates a sequence of `n` numbers, spaced between `start` and `end` on a logarithmic scale,
  where `end` is excluded from the range."
  [start end n]
  (if (= start end)
    `(~start)
    (->> (linspace (math/log start) (math/log end) n)
         (map #(math/exp %))
         (take n))))

(defn log-normalize
  "Normalize a vector of log probabilities, while staying in the log domain."
  [weights]
  (if (map? weights)
    (let [z (logsumexp (vals weights))]
      (reduce-kv (fn [m k v]
                   (assoc m k (- v z)))
                 {}
                 weights))
    (let [z (logsumexp weights)]
      (map #(- % z) weights))))

(defn log-diff
  "Exp, take the difference between to values, then return to the log domain."
  [a b]
  (math/log (- (math/exp a) (math/exp b))))
