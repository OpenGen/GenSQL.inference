(ns inferenceql.inference.gpm.crosscat-test
  (:require [inferenceql.inference.test-models.crosscat :refer [model]]
            [clojure.test :refer [deftest is]]
            [inferenceql.inference.gpm :as gpm]))

(deftest simulate
  (let [sim-no-constraint (gpm/simulate model [:color :height :flip] {})
        sim-constraint-not-target (gpm/simulate model [:color] {:height 0.4})
        sim-constraints-are-target (gpm/simulate model [:color :height] {:color "blue" :height 0.4})]
    ;; Simply checking that we generated all columns without error.
    (is (= #{:color :height :flip} (set (keys sim-no-constraint))))
    (is (= #{:color} (set (keys sim-constraint-not-target))))
    (is (= "blue" (:color sim-constraints-are-target)))
    (is (= 0.4 (:height sim-constraints-are-target)))))

(deftest logpdf
  (let [no-constraints (gpm/logpdf model {:color "red" :height 4.0 :flip true} {})
        constraints-match-target (gpm/logpdf model {:color "red" :height 4.0} {:color "red" :height 4.0})
        mistmatch (gpm/logpdf model
                              {:color "red" :height 4.0 :flip true} {:color "blue" :height 4.0 :flip true})
        match-subset (gpm/logpdf model
                                 {:color "red" :height 4.0} {:color "red" :flip true})
        nonmatch-subset (gpm/logpdf model
                                    {:height 4.0} {:color "red" :flip true})
        no-target (gpm/logpdf model
                              {} {:color "red" :height 4.0 :flip true})
        fully-constrained-target (gpm/logpdf model
                                             {:color "red" :height 4.0} {:color "red" :height 4.0 :flip true})]
    (is (number? no-constraints))
    (is (= 0.0 constraints-match-target))
    (is (= ##-Inf mistmatch))
    (is (= match-subset nonmatch-subset))
    (is (= no-target fully-constrained-target 0.0))))

(deftest variables
  (is (= #{:color :height :flip} (gpm/variables model))))
