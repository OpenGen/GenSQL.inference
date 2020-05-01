(ns inferenceql.inference.multimixture.utils)

(defn prun [n f]
  "Runs `n` parallel calls to function `f`, that is assumed to have
  no arguments."
  #?(:clj (apply pcalls (repeat n f))
     :cljs (repeatedly n f)))

(defn transpose
  "Applies the standard tranpose operation to a collection. Assumes that
  `coll` is an object capable of having a transpose."
  [coll]
  (apply map vector coll))
