(ns inferenceql.inference.gpm.primitive-gpms-test
  (:require [clojure.test :as test :refer [deftest is]]
            [inferenceql.inference.gpm.primitive-gpms :as primitives]))

(deftest primitive?
  (let [bernoulli-name "flip"
        bernoulli-primitive :bernoulli
        bernoulli-suff-stats {:n 1 :x-sum 1}
        bernoulli-hyperparameters {:alpha 1.5 :beta 0.5}

        categorical-name "choose"
        categorical-primitive :categorical
        categorical-options ["a" "b" "c"]
        categorical-suff-stats {:n 3 :counts {"a" 1 "b" 0 "c" 2}}
        categorical-hyperparameters {:alpha 2}

        gaussian-name "bell"
        gaussian-primitive :gaussian
        gaussian-suff-stats {:n 3 :sum-x 6 :sum-x-sq 14}
        gaussian-hyperparameters {:m 0 :r 1 :s 1 :nu 1}]
    (is (and (primitives/primitive? (primitives/->pGPM bernoulli-primitive bernoulli-name))
             (primitives/primitive? (primitives/->pGPM bernoulli-primitive bernoulli-name :suff-stats bernoulli-suff-stats))
             (primitives/primitive? (primitives/->pGPM bernoulli-primitive
                                                       bernoulli-name
                                                       :suff-stats bernoulli-suff-stats
                                                       :hyperparameters bernoulli-hyperparameters))))

    (is (and (primitives/primitive? (primitives/->pGPM categorical-primitive categorical-name :suff-stats categorical-suff-stats))
             (primitives/primitive? (primitives/->pGPM categorical-primitive categorical-name :options categorical-options))
             (primitives/primitive? (primitives/->pGPM categorical-primitive
                                                       categorical-name
                                                       :suff-stats categorical-suff-stats
                                                       :hyperparameters categorical-hyperparameters))))

    (is (and (primitives/primitive? (primitives/->pGPM gaussian-primitive gaussian-name))
             (primitives/primitive? (primitives/->pGPM gaussian-primitive gaussian-name :suff-stats gaussian-suff-stats))
             (primitives/primitive? (primitives/->pGPM gaussian-primitive
                                                       gaussian-name
                                                       :suff-stats gaussian-suff-stats
                                                       :hyperparameters gaussian-hyperparameters))))
    (is (not (primitives/primitive? {bernoulli-name "false"})))
    (is (thrown? #?(:clj Exception :cljs js/Error)
                 (primitives/->pGPM :bad-bernoulli bernoulli-name)))))
