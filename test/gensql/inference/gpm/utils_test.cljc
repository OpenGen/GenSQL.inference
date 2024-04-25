(ns gensql.inference.gpm.utils-test
  (:require
    [clojure.math :as math]
    [clojure.test :refer [deftest is]]
    [gensql.inference.utils :as utils]))

(def logs [-1.0 -2.0 -3.0])
(def weights [-4.0 -5.0 -6.0])

(defn is-almost-equal? [a b] (utils/almost-equal? a b utils/relerr 0.0000000001))

(deftest log-sum-exp
  (is (= (utils/logsumexp logs) (math/log (apply + (map math/exp logs))))))

(deftest log-mean-exp
  (is (is-almost-equal? (utils/logmeanexp logs) (math/log (utils/average (map math/exp logs))))))

(deftest log-weighted-mean-exp
  (is (is-almost-equal? (utils/logmeanexp-weighted weights logs)
                        (math/log (/ (apply + (map *
                                                   (map math/exp weights)
                                                   (map math/exp logs)))
                                     (apply + (map math/exp weights))))))
  (let [w -0.6
        l -0.7]
    (is (is-almost-equal? (utils/logmeanexp-weighted (list w w)
                                                     (list l l))
                          l))
    (is (is-almost-equal? (utils/logmeanexp-weighted [w w]
                                                     [l l])
                          l))))
