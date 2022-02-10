(ns inferenceql.inference.gpm
  (:refer-clojure :exclude [merge read read-string])
  (:require [clojure.core :as clojure]
            [clojure.edn :as edn]
            [inferenceql.inference.gpm.column :as column]
            [inferenceql.inference.gpm.compositional :as compositional]
            [inferenceql.inference.gpm.crosscat :as xcat]
            #?(:clj [inferenceql.inference.gpm.ensemble :as ensemble])
            #?(:clj [inferenceql.inference.gpm.http :as http])
            [inferenceql.inference.gpm.multimixture :as mmix]
            [inferenceql.inference.gpm.multimixture.specification :as mmix.spec]
            [inferenceql.inference.gpm.primitive-gpms.bernoulli :as bernoulli]
            [inferenceql.inference.gpm.primitive-gpms.categorical :as categorical]
            [inferenceql.inference.gpm.primitive-gpms.gaussian :as gaussian]
            [inferenceql.inference.gpm.proto :as gpm-proto]
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

(defn insert
  "Given a non-parametric GPM and it's partition, insert a row into into the correct
  category (aka the correct table/cluster)"
  [gpm values]
  (gpm-proto/insert gpm values))

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

(defn condition
  "Conditions the provided generative probabilistic model such that it only
  simulates the provided targets, and is always subject to the provided
  conditions."
  [gpm conditions]
  (gpm-proto/condition gpm conditions))

(defn constrain
  "Constrains a GPM by an event. event is a tree-like data structure. opts is a
  collection of functions for traversal of that tree-like data structure. Nodes
  in that data structure are either operations (which can have child nodes),
  variables, or values.

  Required keys for opts include:
    - :operation? must be a fn of one arg that returns true if its argument is
      an operation node
    - :operands must be a fn of one arg that returns the arguments to an
      operation node
    - :operator must be a fn of one arg that returns the operator for an
      operation node
    - :variable? must be a fn of one arg that returns true if its argument is a
      variable"
  [gpm event opts]
  (gpm-proto/constrain gpm event opts))

(def readers
  {'inferenceql.inference.gpm.column.Column column/map->Column
   'inferenceql.inference.gpm.compositional.Compositional compositional/map->Compositional
   'inferenceql.inference.gpm.crosscat.XCat xcat/map->XCat
   #?@(:clj ['inferenceql.inference.gpm.ensemble.Ensemble ensemble/map->Ensemble])
   'inferenceql.inference.gpm.multimixture.Multimixture mmix/map->Multimixture
   'inferenceql.inference.gpm.primitive_gpms.bernoulli.Bernoulli bernoulli/map->Bernoulli
   'inferenceql.inference.gpm.primitive_gpms.categorical.Categorical categorical/map->Categorical
   'inferenceql.inference.gpm.primitive_gpms.gaussian.Gaussian gaussian/map->Gaussian
   'inferenceql.inference.gpm.view.View view/map->View})

(defn as-gpm
  "Coerce argument to a value that implements `gpm-proto.GPM`."
  [x]
  (cond (satisfies? gpm-proto/GPM x) x
        (mmix.spec/spec? x) (Multimixture x)
        #?@(:clj [(let [url? #(re-find #"^https?://" %)]
                    (url? x))
                  (http x)])
        :else (throw (ex-info "Cannot coerce value to GPM" {:value x}))))

(defn read
  "Like `clojure.edn/read` but includes readers for records in
  `inferneceql.inference`."
  ([stream]
   (read {} stream))
  ([opts stream]
   (as-gpm
    (edn/read (update opts :readers clojure/merge readers)
              stream))))

(defn ^:export read-string
  "Like `clojure.edn/read-string` but includes readers for records from
  `inferenceql.inference`."
  ([s]
   (read-string {} s))
  ([opts s]
   (as-gpm
    (edn/read-string (update opts :readers clojure/merge readers)
                     s))))

(defn variables
  "Given a GPM, returns the variables it supports."
  [gpm]
  (gpm-proto/variables gpm))

(defn logprob
  "Returns the log probability of of an event under a model."
  [gpm event]
  (gpm-proto/logprob gpm event))
