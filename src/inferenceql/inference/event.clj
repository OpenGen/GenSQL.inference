(ns inferenceql.inference.event
  (:require [clojure.core.match :as match]))

(defn operator
  [expr]
  (first expr))

(defn children
  [expr]
  (rest expr))

(defn simple?
  [event]
  (and (contains? #{'> '<} (operator event))
       (= 3 (count event))
       (some number? event)))

(defn negate
  [event]
  (match/match event
    (['> a b] :seq) (list '< b a)
    (['< a b] :seq) (list '> b a)))

(defn dnf?
  [event]
  (let [and? (fn [event]
               (and (= 'and (operator event))
                    (> (count (children event))
                       1)))
        children (children event)]
    (and (= 'or (operator event))
         (or (every? and? children)
             (and (= 1 (count children))
                  (simple? (first children)))))))

(comment

  (dnf? '(or (> x 3)))
  (dnf? '(or (and (> x 3))))
  (dnf? '(or (and (> x 3)
                  (> x 3))))

  (negate '[> x 0])
  (negate '(> x 0))
  (negate '(< x 0))

  ,)
