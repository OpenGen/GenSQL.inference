(ns inferenceql.inference.inference
  (:require [inferenceql.inference.kernels.category :as category]
            [inferenceql.inference.kernels.hyperparameters :as col-hypers]
            [inferenceql.inference.kernels.concentration-hypers :as conc-hypers]
            [inferenceql.inference.kernels.view :as view]))

(defn infer-dpmm
  [gpm n]
  (reduce (fn [gpm' _]
            (-> gpm'
                category/infer
                col-hypers/infer
                conc-hypers/infer))
          gpm
          (range n)))

(defn infer-xcat
  [gpm n]
  (reduce (fn [gpm' _]
            (-> gpm'
                category/infer
                col-hypers/infer
                conc-hypers/infer
                view/infer))
          gpm
          (range n)))
