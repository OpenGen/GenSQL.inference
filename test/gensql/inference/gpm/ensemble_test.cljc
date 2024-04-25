(ns gensql.inference.gpm.ensemble-test
  (:require [gensql.inference.test-models.crosscat :as crosscat]
            [clojure.test :refer [deftest is]]
            [gensql.inference.gpm :as gpm]
            [gensql.inference.gpm.ensemble :as ensemble]
            [gensql.inference.gpm.conditioned :as conditioned]))

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
  (let [x-and-y (gpm/logpdf models {:color "red" :height 0.4} {})
        x-given-y (gpm/logpdf models {:color "red"} {:height 0.4})
        y (gpm/logpdf models {:height 0.4} {}) ]
    (is (number? x-and-y))
    (is (number? x-given-y))
    (is (number? y))
    ;; test Bayes' Rule
    (is (= (+ x-given-y y) x-and-y)))))

#?(:clj (deftest condition
  (let [x-and-y (gpm/logpdf models {:color "red" :height 0.4} {})
        x-given-y (gpm/logpdf (gpm/condition models {:height 0.4}) {:color "red"} {})
        x-given-y-indep-z (gpm/logpdf (gpm/condition (gpm/condition models {:height 0.4}) {:flip true}) {:color "red"} {})
        x-given-indep-z-y (gpm/logpdf (gpm/condition (gpm/condition models {:flip true}) {:height 0.4}) {:color "red"} {})
        y (gpm/logpdf models {:height 0.4} {})]
    (is (number? x-and-y))
    (is (number? x-given-y))
    (is (number? y))
    ;; test Bayes' Rule
    (is (= (+ x-given-y y) x-and-y))
    (is (= (+ x-given-y-indep-z y) x-and-y))
    (is (= (+ x-given-indep-z-y y) x-and-y)))))

(deftest variables
  (is (= #{:color :height :flip} (gpm/variables models))))
