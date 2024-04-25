(ns gensql.inference.js-test
  (:require [cljs-bean.core :as bean]
            [clojure.test :refer [deftest is]]
            [gensql.inference.js :as inference.js]
            [gensql.inference.test-models.crosscat :refer [model]]))

(deftest simulate
  (let [actual (bean/->clj (inference.js/simulate model
                                                  #js ["color" "height" "flip"]
                                                  #js {}))]
    (is (= (set (keys actual))
           #{:color :height :flip}))))

(deftest logpdf
  (let [actual (bean/->clj (inference.js/logpdf model
                                                #js {"color" "red" "height" 4.0 "flip" true}
                                                #js {}))]
    (is (number? actual))))

(deftest variables
  (let [actual (bean/->clj (inference.js/variables model))]
    (is (= #{:color :height :flip}
           (set (map keyword actual))))))
