(ns inferenceql.inference.multimixture.metrics-viz
  (:require [clojure.spec.alpha :as s]
            [clojure.string :refer [split]]
            [inferenceql.inference.multimixture :as mmix]
            [inferenceql.inference.multimixture.metrics :as metrics]
            [inferenceql.inference.multimixture.specification :as spec]
            [clojure.data.json :as json]
            [zane.vega.repl :refer [vega]]))

(defmacro with-out-str-data-time-map
  "Returns time and data of an expression of the form `(time expression)`
  Returns a map with keys `:result` and `:time`.
  Returns `:time` in milliseconds.
  If the expression results in a lazy sequence, you must pass
  `(time (doall expression))` instead."
  [& body]
  `(let [s# (new java.io.StringWriter)]
     (binding [*out* s#]
       (let [r# ~@body]
         {:result r#
          :time (Float/parseFloat (nth (split (str s#) #" ")
                                       2))}))))

(defn generate-time-data
  "Generates `n` data samples from the distribution `dist`, and calculates the
  provided `metric`."
  [metric dist n]
  (let [dims (count dist)
        time-data (with-out-str-data-time-map (time (metrics/generate-categorical-data dist n)))
        timestamp (get time-data :time)
        generated-data (get time-data :result)
        iters (count generated-data)
        metric-result (metric dist (metrics/get-empirical-distribution
                                    generated-data
                                    dims))]
    {:time timestamp
     :metric-result metric-result
     :iters iters}))

(defn generate-time-data-no-metric
  "Generates `n` data samples from the distribution `dist`. "
  [dist n]
  (let [dims (count dist)
        time-data (with-out-str-data-time-map (time (metrics/generate-categorical-data dist n)))
        timestamp (get time-data :time)
        generated-data (get time-data :result)
        iters (count generated-data)
        emp-dist (metrics/get-empirical-distribution generated-data dims)]
    {:emp-dist emp-dist
     :time timestamp
     :iters iters}))

(defn generate-time-data-over-range-no-metric
  "Generates time data as defined in `generate-time-data` over the specified
  `n-range`. Returns a fully realized sequence."
  [dist n-range]
  (mapv (partial generate-time-data-no-metric dist) n-range))

(defn generate-time-data-over-range
  "Generates time data as defined in `generate-time-data` over the specified
  `n-range`. Returns a fully realized sequence."
  [metric dist n-range]
  (mapv (partial generate-time-data metric dist) n-range))

(defn generate-metrics-spec
  "Generates a spec to be used with Vega for a specified `metric`, `dist`, and `n-range`
  of iterations."
  [metric dist n-range title]
  (let [data (vec (generate-time-data-over-range metric dist n-range))]
    {:$schema "https://vega.github.io/schema/vega-lite/v4.json"
     :description "A simple bar chart with embedded data."
     :data {:values data}
     :title title
     :mark {:type "line"
            :interpolate "monotone"}
     :encoding {:x {:field :time :type "quantitative"}
                :y {:field :metric-result :type "quantitative"}}}))

(defn process-time-data-with-metric
  "Maps output of time-datageneration to a spec form usable in vega-lite."
  [metric dist time-data metric-title dist-title]
  (mapv (fn [time-datum]
          {"Metric Result" (metric dist (get time-datum :emp-dist))
           :time (get time-datum :time)
           :metric-title metric-title
           :dist-title dist-title})
        time-data))

(defn generate-aggregate-spec
  "Generates a spec comparing a number of metrics on `dist 1` and `dist 2` over time."
  [dist-1 dist-2 n-range title]
  (let [dist-1-data-no-metric (generate-time-data-over-range-no-metric
                               dist-1 n-range)
        dist-2-data-no-metric (generate-time-data-over-range-no-metric
                               dist-2 n-range)
        kl-title "Kullback-Leibler Divergence"
        tv-title "Total Variational Distance"
        js-title "Jensen-Shannon Distance"
        dist-1-title "Bernoulli Variable"
        dist-2-title "Categorical Variable"
        all-data (concat
                  (process-time-data-with-metric
                   metrics/kl-divergence dist-1 dist-1-data-no-metric kl-title dist-1-title)
                  (process-time-data-with-metric
                   metrics/tv-distance dist-1 dist-1-data-no-metric tv-title dist-1-title)
                  (process-time-data-with-metric
                   metrics/jensen-shannon-divergence dist-1 dist-1-data-no-metric js-title dist-1-title)
                  (process-time-data-with-metric
                   metrics/kl-divergence dist-2 dist-2-data-no-metric kl-title dist-2-title)
                  (process-time-data-with-metric
                   metrics/tv-distance dist-2 dist-2-data-no-metric tv-title dist-2-title)
                  (process-time-data-with-metric
                   metrics/jensen-shannon-divergence dist-2 dist-2-data-no-metric js-title dist-2-title))]
    {:$schema "https://vega.github.io/schema/vega-lite/v4.json"
     :data {:values all-data}
     :vconcat [{:mark "point"
                :transform [{:filter "datum['dist-title'] === 'Bernoulli Variable'"}]
                :encoding {:facet {:field "metric-title"
                                   :type "nominal"
                                   :columns 3
                                   :title dist-1-title}
                           :x {:field :time
                               :axis {:title "Time (ms)"}
                               :type "quantitative"}
                           :y {:field "Metric Result"
                               :type "quantitative"}
                           :color {:field :metric-title
                                   :type "nominal"
                                   :legend nil}}}
               {:mark "point"
                :transform [{:filter "datum['dist-title'] === 'Categorical Variable'"}]
                :encoding {:facet {:field "metric-title"
                                   :type "nominal"
                                   :columns 3
                                   :title dist-2-title}
                           :x {:field :time
                               :axis {:title "Time (ms)"}
                               :type "quantitative"}
                           :y {:field "Metric Result"
                               :type "quantitative"}
                           :color {:field :metric-title
                                   :type "nominal"
                                   :legend nil}}}]}))


(defn generate-metrics-viz
  ([metric dist n-range title]
   (let [spec (generate-metrics-spec metric dist n-range title)]
     (vega spec))))

(defn generate-aggregate-metrics-viz
  [dist-1 dist-2 n-range title]
  (let [spec (generate-aggregate-spec dist-1 dist-2 n-range title)]
    (vega spec)))
