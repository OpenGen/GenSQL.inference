(ns inferenceql.inference.gpm-test
  (:require [clojure.test :refer [deftest is]]
            [inferenceql.inference.gpm :as gpm]))

(deftest smoke
  (let [spec {:vars {:x :gaussian}
              :views [[{:probability 1
                        :parameters {:x {:mu 0 :sigma 1}}}]]}]
    (is (gpm/gpm? (gpm/Multimixture spec)))))
