(ns inferenceql.inference.gpm.ensemble-test
  (:require [inferenceql.inference.test-models.crosscat :as crosscat]
            [clojure.test :refer [deftest is]]
            [inferenceql.inference.gpm :as gpm]
            [inferenceql.inference.gpm.ensemble :as ensemble]))

(def models (ensemble/ensemble [crosscat/model crosscat/model]))

#?(:clj (deftest simulate
  (let [sim-no-constraint (gpm/simulate models [:color :height :flip] {})
        sim-constraint-not-target (gpm/simulate models [:color] {:height 0.4})
        sim-constraints-are-target (gpm/simulate models [:color :height] {:color "blue" :height 0.4})]
    ;; Simply checking that we generated all columns without error.
    (is (= #{:color :height :flip} (set (keys sim-no-constraint))))
    (is (= #{:color} (set (keys sim-constraint-not-target))))
    (is (= "blue" (:color sim-constraints-are-target)))
    (is (= 0.4 (:height sim-constraints-are-target))))))

#?(:clj (deftest logpdf
  (let [x-and-y (gpm/logpdf models {:color "red" :flip true} {})
        x-given-y (gpm/logpdf models {:color "red"} {:flip true})
        y (gpm/logpdf models {:flip true} {}) ]
    (is (number? x-and-y))
    (is (number? x-given-y))
    (is (number? y))
    ;; test Bayes' Rule
    (is (= (+ x-given-y y) x-and-y)))))

(deftest variables
  (is (= #{:color :height :flip} (gpm/variables models))))
