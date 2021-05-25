(ns inferenceql.inference.js
  "Functions from the gpm namespace for use from Javascript."
  (:require [inferenceql.inference.gpm :as gpm]
            [cljs-bean.core :refer [->clj]]))

(defn ^:export isGPM
  "Returns `true` if `x` is a generative population model."
  [x]
  (gpm/gpm? x))

(defn ^:export variables
  "Given a GPM, returns the variables it supports."
  [gpm]
  (clj->js (gpm/variables gpm)))

(defn ^:export simulate
  "Given a GPM, simulates a sample of the variables in `targets` given `constraints`."
  [gpm targets constraints]
  (clj->js (gpm/simulate gpm
                         (map keyword (->clj targets))
                         (->clj constraints))))

(defn ^:export logpdf
  "Given a GPM, calculates the logpdf of `targets` given `constraints`."
  [gpm targets constraints]
  (clj->js (gpm/logpdf gpm (->clj targets) (->clj constraints))))

(defn ^:export incorporate
  "Given a GPM, incorporates values into the GPM by updating its sufficient statistics."
  [gpm values]
  (gpm/incorporate gpm (->clj values)))

(defn ^:export condition
  "Conditions the provided generative probabilistic model such that it only
  simulates the provided targets, and is always subject to the provided
  conditions."
  [gpm conditions]
  (gpm/condition gpm (->clj conditions)))

(defn ^:export constrain
  "Constrains a GPM by an event. event is a tree-like data structure. opts is a
  collection of functions for traversal of that tree-like data structure. Nodes
  in that data structure are either operations (which can have child nodes),
  variables, or values.

  Required keys for opts include:
  (these keys can be strings as well as keywords)
    - :operation? must be a fn of one arg that returns true if its argument is
      an operation node
    - :operands must be a fn of one arg that returns the arguments to an
      operation node
    - :operator must be a fn of one arg that returns the operator for an
      operation node
    - :variable? must be a fn of one arg that returns true if its argument is a
      variable"
  [gpm event opts]
  (gpm/constrain gpm event (->clj opts)))
