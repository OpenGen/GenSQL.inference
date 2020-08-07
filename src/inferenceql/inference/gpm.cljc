(ns inferenceql.inference.gpm
  (:require [inferenceql.inference.gpm.multimixture :as mmix]
            #?(:clj [inferenceql.inference.gpm.http :as http])
            [inferenceql.inference.gpm.proto :as gpm-proto]
            [inferenceql.inference.gpm.column :as column]
            [inferenceql.inference.gpm.view :as view]))

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

(defn Column
  "Returns a CrossCat Column GPM.

  It is the responsibility of the user to make sure that all categories
  are of the correct type, and assignments are consistent in terms of assigning
  a particular value to a given category.

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
  (column/->Column var-name stattype categories assignments hyperparameters hyper-grid metadata))

(defn View
  "Returns a CrossCat View GPM.

  It is the responsibility of the user to make sure that all categories
  are of the correct type, and assignments are consistent in terms of assigning
  a particular value to a given category.

  columns: map of {column-symbol column}, where each column is a proper Column GPM.
  latents: map of the below structure, used to keep track of row-category assignments,
  as well as category sufficient statistics:

  {:alpha  number                     The concentration parameter for the Column's CRP
  :counts {category-name count}      Maps category name to size of the category. Updated
  incrementally instead of being calculated on the fly.
  :y {row-identifier category-name}  Maps rows to their current category assignment.

  assignments: map of {value {:row-ids #{row-ids} category-symbol count}}, used for (un)incorporating
  by value alone. The :row-ids key for each set of values is used for CrossCat inference
  and the internal labeling of the data."

  [columns latents assignments]
  (view/->View columns latents assignments))

(defn dpmm
  "Returns a CrossCat View GPM, given a View specification, latent assignments of rows
  to categories, variable types, options (for categorical variables) and the corresponding data."
  ([{:keys [model latents types data options]}]
   (dpmm model latents types data {:options options :crosscat true}))
  ([model latents types data]
   (dpmm model latents types data {:options {} :crosscat true}))
  ([model latents types data {:keys [options crosscat]}]
   (view/construct-view-from-latents model latents types data {:options options :crosscat crosscat})))

(defn Multimixture
  "Wrapper to provide conversion to Multimixture model."
  [model]
  (mmix/map->Multimixture model))

(defn gpm?
  "Returns `true` if `x` is a generative population model."
  [x]
  (satisfies? gpm-proto/GPM x))

(defn incorporate
  "Given a GPM, incorporates values into the GPM by updating its sufficient statistics."
  [gpm values]
  (gpm-proto/incorporate gpm values))

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
  "Given a GPM, simulates a sample of the variables in `targets` given `constraints`."
  [gpm targets constraints]
  (gpm-proto/simulate gpm targets constraints))
