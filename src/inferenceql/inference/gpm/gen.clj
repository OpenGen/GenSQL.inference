(ns inferenceql.inference.gpm.gen
  (:require [gen.dynamic :as dynamic]
            [gen.dynamic.choice-map :as choice-map]
            [gen.generative-function :as gf]
            [gen.inference.importance :as importance]
            [gen.trace :as trace]
            [inferenceql.inference.gpm :as gpm]
            [inferenceql.inference.gpm.proto :as gpm.proto])
  (:import [gen.dynamic DynamicDSLFunction]))

(def ^:dynamic *n-samples* 1E3)

(extend-type DynamicDSLFunction
  gpm.proto/GPM
  (simulate [gf targets constraints]
    (-> (importance/resampling gf [] (choice-map/choice-map constraints) *n-samples*)
        (:trace)
        (trace/choices)
        (choice-map/unwrap)
        (select-keys targets)))

  (logpdf [gf targets constraints]
    (let [log-marginal-likelihood #(-> (importance/resampling gf [] (choice-map/choice-map %) *n-samples*)
                                       (:weight))]
      (if (empty? constraints)
        (log-marginal-likelihood targets)
        (- (log-marginal-likelihood (merge targets constraints))
           (log-marginal-likelihood constraints))))))

(comment

  (require '[gen.distribution.commons-math :as commons-math]
           '[gen.dynamic :as dynamic :refer [gen]]
           '[gen.trace :as trace]
           '[inferenceql.inference.sample :as sample])

  (importance/resampling sample/generate-row [] (choice-map/choice-map) 100)

  (-> (gf/generate sample/generate-row [])
      (:trace)
      (trace/choices))

  (def coin-model
    (gen []
      (if (dynamic/trace! :flip commons-math/bernoulli 0.5)
        (dynamic/trace! :outcome commons-math/bernoulli 0.99)
        (dynamic/trace! :outcome commons-math/bernoulli 0.01))))

  (Math/exp (gpm/logpdf coin-model {:outcome true} {:flip true}))
  (Math/exp (gpm/logpdf coin-model {:flip true} {:outcome true}))

  (Math/exp (gpm/logpdf coin-model {:outcome false} {:flip false}))
  (Math/exp (gpm/logpdf coin-model {:flip false} {:outcome false}))

  (Math/exp (gpm/logpdf coin-model {:flip true} {:outcome false}))
  (Math/exp (gpm/logpdf coin-model {:flip false} {:outcome true}))

  (gpm/simulate sample/generate-row (gpm/variables sample/generate-row) {})
  (gpm/simulate sample/generate-row #{"Period_minutes"} {})
  (gpm/simulate sample/generate-row #{"Period_minutes"} {"Period_minutes" 1000})
  (gpm/simulate sample/generate-row #{"Perigee_km"} {"Apogee_km" 35800})

  (Math/exp (gpm/logpdf sample/generate-row {"Perigee_km" 35800} {"Apogee_km" 35800}))
  (Math/exp (gpm/logpdf sample/generate-row {"Perigee_km" 35800} {}))

  (gpm/variables (with-meta {:some-value "for example"}
                   {`gpm.proto/variables (constantly #{:a :b :c})}))

  ,)
