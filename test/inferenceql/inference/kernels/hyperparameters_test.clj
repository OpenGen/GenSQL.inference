(ns inferenceql.inference.kernels.hyperparameters-test
  (:require [clojure.test :as test :refer [deftest is]]
            [inferenceql.inference.kernels.hyperparameters :as hypers]
            [inferenceql.inference.gpm.column :as column]))

(def hypers-bernoulli
  {:alpha 0.5 :beta 0.5})

(def hypers-categorical
  {:alpha 1})

(def hypers-gaussian
  {:m 0 :r 1 :s 1 :nu 1})

(def data-bernoulli-high-alpha
  [true true true true true true])

(def data-bernoulli-high-beta
  [false false false false false false])

(def data-categorical-low-alpha
  ["red" "red" "red" "green" "green" "green"])

(def data-categorical-high-alpha
  ["red" "green" "blue" "red" "green" "blue"])

(def data-gaussian
  [4 4 4 6 6 6])

(def latents
  {:alpha 1
   :counts {:one 3 :two 3}
   :y {0 :one
       1 :one
       2 :one
       3 :two
       4 :two
       5 :two}})

(def column-bernoulli-high-alpha
  (column/construct-column-from-latents "flip"
                                        :bernoulli
                                        hypers-bernoulli
                                        latents
                                        (into {} (map-indexed vector data-bernoulli-high-alpha))))

(def column-bernoulli-high-beta
  (column/construct-column-from-latents "flip"
                                        :bernoulli
                                        hypers-bernoulli
                                        latents
                                        (into {} (map-indexed vector data-bernoulli-high-beta))))

(def column-categorical-low-alpha
  (column/construct-column-from-latents "color"
                                        :categorical
                                        hypers-categorical
                                        latents
                                        (into {} (map-indexed vector data-categorical-low-alpha))
                                        {:options {"color" ["red" "blue" "green"]}}))

(def column-categorical-high-alpha
  (column/construct-column-from-latents "color"
                                        :categorical
                                        hypers-categorical
                                        latents
                                        (into {} (map-indexed vector data-categorical-high-alpha))
                                        {:options {"color" ["red" "blue" "green"]}}))

(def column-gaussian
  (column/construct-column-from-latents "height"
                                        :gaussian
                                        hypers-gaussian
                                        latents
                                        (into {} (map-indexed vector data-gaussian))))

;; Tests column hyperparameter inference on a Bernoulli pGPM.
(deftest bernoulli-hypers
  (let [n-columns 1000
        columns-high-alpha (->> #(hypers/infer column-bernoulli-high-alpha)
                                (repeatedly n-columns)
                                (map #(:hyperparameters %))
                                (apply merge-with + {})
                                (reduce-kv (fn [m k v]
                                             (assoc m k (/ v n-columns)))
                                           {}))
        columns-high-beta (->> #(hypers/infer column-bernoulli-high-beta)
                               (repeatedly n-columns)
                               (map #(:hyperparameters %))
                               (apply merge-with + {})
                               (reduce-kv (fn [m k v]
                                            (assoc m k (/ v n-columns)))
                                          {}))]
    (is (> (:alpha columns-high-alpha) 0.75))
    (is (< (:beta columns-high-alpha) 1.5))
    (is (< (:alpha columns-high-beta) 1.5))
    (is (> (:beta columns-high-beta) 0.75))))

;; Tests column hyperparameter inference on a Categorical pGPM.
(deftest categorical-hypers
  (let [n-columns 1000
        columns-low-alpha (->> #(hypers/infer column-categorical-low-alpha)
                                (repeatedly n-columns)
                                (map #(:hyperparameters %))
                                (apply merge-with + {})
                                (reduce-kv (fn [m k v]
                                             (assoc m k (/ v n-columns)))
                                           {}))
        columns-high-alpha (->> #(hypers/infer column-categorical-high-alpha)
                                (repeatedly n-columns)
                                (map #(:hyperparameters %))
                                (apply merge-with + {})
                                (reduce-kv (fn [m k v]
                                             (assoc m k (/ v n-columns)))
                                           {}))]
    (is (< (:alpha columns-low-alpha) 2.75))
    (is (> (:alpha columns-high-alpha) 2.75))))

;; Tests column hyperparameter inference on a Gaussian pGPM.
(deftest gaussian-hypers
  (let [n-columns 1000
        columns (->> #(hypers/infer column-gaussian)
                     (repeatedly n-columns)
                     (map #(:hyperparameters %))
                     (apply merge-with + {})
                     (reduce-kv (fn [m k v]
                                  (assoc m k (/ v n-columns)))
                                {}))]
    (is (> (:m columns) 4))
    (is (> (:r columns) 0.5))
    (is (< (:s columns) 1))
    (is (> (:nu columns) 0.5))))
