(ns inferenceql.inference.inference
  (:require [progrock.core :as prog]
            [inferenceql.inference.gpm.view :as view]
            [inferenceql.inference.gpm.crosscat :as xcat]
            [inferenceql.inference.kernels.category :as category]
            [inferenceql.inference.kernels.hyperparameters :as col-hypers]
            [inferenceql.inference.kernels.concentration-hypers :as conc-hypers]
            [inferenceql.inference.kernels.view :as k.view]))

(defn pr-status-bar
  [bar]
  (prog/print bar {:format ":progress/:total  :percent% [:bar] ETA: :remaining, Elapsed: :elapsed"}))

(defn infer
  ([gpm n]
   (infer gpm n (prog/progress-bar n)))
  ([gpm n bar]
   (assert (or (view/view? gpm)
               (xcat/xcat? gpm))
           (str "GPM must either be View or XCat: " (type gpm)))
   (pr-status-bar bar)
   (let [[final-gpm final-bar] (reduce (fn [[gpm' bar'] _]
                                         (let [inferred (-> gpm'
                                                            category/infer
                                                            col-hypers/infer
                                                            conc-hypers/infer
                                                            (#(if (xcat/xcat? %)
                                                                (k.view/infer %)
                                                                %)))
                                               new-bar (prog/tick bar')]
                                           (pr-status-bar new-bar)
                                           [inferred
                                            new-bar]))
                                       [gpm bar]
                                       (range n))]
     (pr-status-bar (prog/done final-bar))
     final-gpm)))
