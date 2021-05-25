(ns inferenceql.inference.gpm.crosscat-test
  (:require [inferenceql.inference.test-models.crosscat :refer [model]]
            [clojure.test :refer [deftest is]]
            [inferenceql.inference.gpm :as gpm]))

(deftest simulate
  (let [row (gpm/simulate model [:color :height :flip] {})]
    ;; Simply checking that we generated all columns without error.
    (is (= (set (keys row)) #{:color :height :flip}))))

(deftest logpdf
  (let [probability (gpm/logpdf model {:color "red" :height 4.0 :flip true} {})]
    ;; Simply checking that we can do logpdf on all columns without error.
    (is (number? probability))))

(deftest variables
  (is (= #{:color :height :flip} (gpm/variables model))))
