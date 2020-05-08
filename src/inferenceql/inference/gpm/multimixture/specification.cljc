(ns inferenceql.inference.gpm.multimixture.specification
  (:require [clojure.spec.alpha :as s]))

(s/def ::alpha pos?)

(s/def ::beta pos?)

(s/def ::beta-parameters (s/keys :req-un [::alpha ::beta]))

(s/def ::mu number?)

(s/def ::sigma
  (s/and number? pos?))

(s/def ::gaussian-parameters
  (s/keys :req-un [::mu ::sigma]))

(defn normalized?
  [xs]
  (== 1 (apply + xs)))

(s/def ::probability #(<= 0 % 1))

(s/def ::probability-vector
  (s/and (s/+ ::probability)
         normalized?))

(s/def ::binary-paramters
  (s/and number?
         #(<= 0 % 1)))

(s/def ::categorical-parameters
  (s/map-of string? float?))

(s/def ::distribution-paremeters
  (s/or ::binary-parameters      ::binary-paramters
        ::categorical-parameters ::categorical-parameters
        ::gaussian-parameters    ::gaussian-parameters))

(s/def ::column string?)

(s/def ::row (s/map-of ::column any?))

(s/def ::rows (s/coll-of ::row))

(s/def ::parameters (s/map-of ::column ::distribution-paremeters))

(def distribution? #{:binary :gaussian :categorical})

(s/def ::distribution distribution?)

(s/def ::vars (s/and #(> (count %) 0)
                     (s/map-of ::column ::distribution)))

(s/def ::cluster (s/keys :req-un [::probability ::parameters]))

(s/def ::clusters (s/coll-of ::cluster))

(s/def ::view ::clusters)

(s/def ::views (s/coll-of ::view))

(s/def ::multi-mixture
  (s/keys :req-un [::vars ::views]))

(s/fdef from-json
  :args (s/cat :json (s/map-of string? any?))
  :ret ::multi-mixture)

(defn from-json
  [{:strs [columns views]}]
  (let [vars (reduce-kv (fn [m k v]
                          (assoc m k (keyword v)))
                        {}
                        columns)
        views (mapv (fn [view]
                      (mapv (fn [cluster]
                              (let [column-parameters (dissoc cluster "p")]
                                {:probability (get cluster "p")
                                 :parameters (reduce-kv (fn [m column parameters]
                                                          (let [stattype (get vars column)]
                                                            (assoc m
                                                                   column
                                                                   (case stattype
                                                                     :gaussian (zipmap [:mu :sigma] parameters)
                                                                     :categorical parameters))))
                                                        {}
                                                        column-parameters)}))
                            view))
                    views)]
    {:vars vars
     :views views}))

(s/fdef cluster-variables
  :args (s/cat :cluster ::cluster)
  :ret (s/coll-of ::column))

(defn- cluster-variables
  [cluster]
  (set (keys (:parameters cluster))))

(s/fdef view-variables
  :args (s/cat :view ::view)
  :ret (s/coll-of ::column))

(defn view-variables
  "Returns the variables assigned to given view."
  [view]
  (cluster-variables (first view)))

(s/fdef variables
  :args (s/cat :mmix ::multi-mixture)
  :ret (s/coll-of ::column))

(defn variables
  "Returns the variables in a multi-mixture."
  [mmix]
  (set (keys (:vars mmix))))

(s/fdef view-index-for-variable
  :args (s/cat :mmix ::multi-mixture
               :variable ::column))

(defn view-index-for-variable
  "Returns the index of the view a given variable was assigned to."
  [mmix variable]
  (some (fn [[i view]]
          (when (contains? (view-variables view) (name variable))
            i))
        (map-indexed vector (:views mmix))))

(defn view-for-variable
  "Returns the view a given variable was assigned to."
  [mmix variable]
  (some (fn [view]
          (when (contains? (:parameters (first view))
                           variable)
            view))
        (:views mmix)))

(defn stattype
  "Returns the statistical type (distribution from `metaprob.distributions`) of a
  variable."
  [mmix variable]
  (get-in mmix [:vars variable]))

(defn nominal?
  "Returns true if `variable` is a nominal variable in `mmix`."
  [mmix variable]
  (= :categorical (stattype mmix variable)))

(defn numerical?
  "Returns true if `variable` is a numerical variable in `multimixture`."
  [mmix variable]
  (= :gaussian (stattype mmix variable)))

(defn parameters
  "Returns the parameters of a variable for a cluster."
  [mmix variable cluster-idx]
  (let [view (view-for-variable mmix variable)]
    (get-in view [cluster-idx :parameters variable])))

(defn mu
  "Returns the mu for the given variable."
  [mmix variable cluster-idx]
  (:mu (parameters mmix variable cluster-idx)))

(defn sigma
  "Returns the sigma for the given variable."
  [mmix variable cluster-idx]
  (:sigma (parameters mmix variable cluster-idx)))

(defn cluster-probability
  [mmix view-idx cluster-idx]
  (get-in mmix [:views view-idx cluster-idx :probability]))

(defn categories
  [mmix variable]
  (-> (view-for-variable mmix variable)
      (get-in [0 :parameters variable])
      keys
      set))

(s/fdef categorical-probabilities
  :args (s/cat :mmix ::multi-mixture
               :variable ::variable
               :cluster-idxs (s/+ nat-int?)))

(defn categorical-probabilities
  "Returns the probabilities for the given categorical variable. If multiple
  clusters are provided the weighted (by cluster probability) sum is returned
  instead."
  ([mmix variable cluster-idx]
   (parameters mmix variable cluster-idx))
  ([mmix variable cluster-idx-1 cluster-idx-2 & more]
   (let [cluster-idxs (into more [cluster-idx-1 cluster-idx-2])
         view-idx (view-index-for-variable mmix variable)
         view (get-in mmix [:views view-idx])]
     (->> cluster-idxs
          (map #(nth view %))
          (map (fn [{:keys [probability parameters]}]
                 (reduce-kv (fn [m k v]
                              (assoc m k (* v probability)))
                            {}
                            (get parameters variable))))
          (apply merge-with +)))))
