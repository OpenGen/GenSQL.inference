(ns inferenceql.inference.event
  "Functions to process SPPL-style events.")

(defn operator
  "Return the operator for an event."
  [expr]
  (first expr))

(defn operator?
  "Returns true of the symbol is a binary operator."
  [s]
  (contains? #{'> '< '= 'not} s))

(defn negated?
  "Returns true if the event is negated."
  [event]
  (= (first event) 'not))

(defn equality?
  "Returns true if the event is an equality event."
  [event]
  (and (=  '= (operator event))
       (= 3 (count event))
       (or (some number? event)
           (some string? event))))

(defn variable?
  "Returns true if symbol is an operator."
  [s]
  (or (symbol? s) (keyword? s)))

(defn third [l] (nth l 2))

(defn eq-event-map
  "Turns an equality event into a map."
  [event]
  (cond
    (or (variable? (second event)) (string? (second event)))
    {(second event) (third event)}
    (or (variable? (third event)) (string? (third event)))
    {(third event) (second event)}))
