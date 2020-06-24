(ns inferenceql.inference.gpm
  (:require [inferenceql.inference.gpm.multimixture :refer [->Multimixture]]
            #?(:clj [inferenceql.inference.gpm.http :as http])
            [inferenceql.inference.gpm.proto :as gpm-proto]
            #?(:clj [inferenceql.inference.gpm.column :as column])))

#?(:clj
   (defn http
     "Returns a generative population model (GPM) that computes results by
  forwarding requests over HTTP. `url` should include the URL scheme and
  authority. Functions that operate on GPMs will trigger HTTP requests to an
  endpoint at `url` with function name as the path. For example, if `url` is
  `http://localhost`, calling `logpdf` will trigger a request to
  `http://localhost/logpdf`. Arguments will be included as JSON in the request
  body as a single JSON object with the argument names being used as keys.
  Responses will be read as JSON with the keys being coerced to keywords."
     [url]
     (http/->HTTP url)))

#?(:clj
   (defn column
     "Returns a CrossCat Column GPM.

     If accessing this constructor directly, it is the responsibility of the user
     to make sure that all categories are of the correct type, and assignments are
     consistent in terms of assigning a particular value to a given category.

     var-name: the name of the variable contained in the column.
     stattype: the statistical type of the variable contained in the column (e.g. :bernoulli).
     categories: a map of {category-symbol category}, where each category must be a pGPM of
                 the Column's statistical type.
     assignments: map of {value {category-symbol count}}, used for (un)incorporating by value alone.
                  Note that identical instances of values are unique only in that `assignments` keeps
                  track of to which categories they belong.
     hyperparameters: the hyperparameters of column; these persist across all categories.
     hyper-grid: a gridded approximation of the hyperparameter space, used in CrossCat inference;
                 this is only updated when values are added or removed to the Column.
     metadata: additional information needed in the column; e.g. for a :categorical Column,
               `metadata` would contain a list of possible values the variable could take."
     [var-name stattype categories assignments hyperparameters hyper-grid metadata]
     (column/->Column var-name stattype categories assignments hyperparameters hyper-grid metadata)))

(defn Multimixture
  "Wrapper to provide conversion to Multimixture model."
  [model]
  (->Multimixture model))

(defn gpm?
  "Returns `true` if `x` is a generative population model."
  [x]
  (satisfies? gpm-proto/GPM x))

(defn logpdf
  "Given a GPM, calculates the logpdf of `targets` given `constraints`."
  [gpm targets constraints]
  (gpm-proto/logpdf gpm targets constraints))

(defn mutual-information
  "Given a GPM, estimates the mutual-information of `target-a` and `target-b`
  given `constraints` with `n-samples`."
  [gpm target-a target-b constraints n-samples]
  (gpm-proto/mutual-information gpm target-a target-b constraints n-samples))

(defn simulate
  "Given a GPM, simulates `n-samples` samples of the variables in `targets`,
  given `constraints`."
  [gpm targets constraints n-samples]
  (gpm-proto/simulate gpm targets constraints n-samples))
