(ns inferenceql.inference.gpm.multimixture.utils)

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
