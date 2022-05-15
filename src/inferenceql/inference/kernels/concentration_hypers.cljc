(ns inferenceql.inference.kernels.concentration-hypers
  (:require #?(:clj [clojure.data.json :as json])
            [clojure.math :as math]
            [inferenceql.inference.gpm.crosscat :as xcat]
            [inferenceql.inference.gpm.view :as view]
            [inferenceql.inference.primitives :as primitives]
            [inferenceql.inference.utils :as utils]))

(defn score-alpha
  "Given sufficient statistics for a CRP, calculates the logpdf for a proposed alpha.
  http://gershmanlab.webfactional.com/pubs/GershmanBlei12.pdf#page=4 (eq 8)"
  [n z k alpha]
  (+ (* k (math/log alpha))
     z
     (primitives/gammaln alpha)
     (- (primitives/gammaln (+ n alpha)))))

(defn infer-alpha
  "Given sufficient statistics for a CRP and list of candidate alphas,
  samples alpha approximately from the posterior."
  [gpm n-grid]
  (let [counts (get-in gpm [:latents :counts])
        n (reduce + (vals counts))
        hyper-grid (utils/log-linspace (/ 1 n) n n-grid)
        z (reduce (fn [acc cnt] (+ acc (primitives/gammaln cnt))) 0 (vals counts))
        k (count counts)
        scores (map #(score-alpha n z k %) hyper-grid)
        normalized-scores (utils/log-normalize scores)
        logps {:p (into {} (map vector hyper-grid normalized-scores))}
        alpha' (primitives/simulate :log-categorical logps)]
    (assoc-in gpm [:latents :alpha] alpha')))

(defn infer-alpha-gamma-prior
  "Assumes a Gamma(1, 1) prior over values of alpha.
  See https://pdfs.semanticscholar.org/004c/1d7d01fca84a0d3102eed92a9f035836e35e.pdf,
  Section 6 for more information.
  This is advantageous when one would like to incorporate the previous value of
  alpha when determining the next value; i.e. this method does not rely on
  Gibbs sampling, and so one is less likely to see large changes to alpha as
  inference proceeds."
  [gpm]
  (let [latents (:latents gpm)
        alpha (:alpha latents)
        counts (:counts latents)
        k (count counts)
        n (reduce + (vals counts))
        a 1
        b 1
        eta (primitives/simulate :beta {:alpha (inc alpha) :beta n})
        pi-eta (/ (+ (dec alpha) k)
                  (+ (dec alpha) k (* n
                                      (- b (math/log eta)))))
        theta (/ 1.0 (- b (math/log eta)))
        sample (fn [k]
                 (primitives/simulate :gamma {:k k
                                      :theta theta}))
        alpha' (if (< (rand) pi-eta)
                 (sample (+ a k))
                 (sample (+ (dec a) k)))]
    (assoc-in gpm [:latents :alpha] alpha')))

;; The below is used for debugging locally, and so does not need to run on the browser.
#?(:clj
   (defn generate-latents-spec
     "Generates Vega spec for visualizing values of alpha for a given CRP,
     under their respective scores. Used only in debugging."
     [normalized-scores]
     (let [data (reduce-kv (fn [m k v]
                             (conj m {"alpha" k "score" v}))
                           []
                           normalized-scores)]
       (json/write-str {"$schema" "https://vega.github.io/schema/vega-lite/v4.json"
                        "data" {"values" data}
                        "mark" "point"
                        "encoding" {
                                    "x" { "field" "alpha"
                                         "type" "quantitative"}
                                    "y" { "field" "score"
                                         "type" "quantitative"}}}))))

(defn infer-alpha-xcat
  "Given a CrossCat model, returns the model with updated concentration
  hyperparameters across and within views."
  [xcat & {:keys [n-grid] :or {n-grid 30}}]
  (let [alpha-g (infer-alpha xcat n-grid)
        alpha-vs (into {} (map (fn [[view-name view]]
                                 {view-name (infer-alpha-gamma-prior view)})
                               (:views xcat)))]
    (-> xcat
        (assoc-in [:latents :alpha] alpha-g)
        ((fn [model]
           (reduce-kv (fn [m view-name alpha]
                        (assoc-in m [:views view-name :latents :alpha] alpha))
                      model
                      alpha-vs))))))

(defn infer
  "Conducts hyperparameter inference on a GPM.
  Supports Column GPMs only."
  ([gpm]
   (-> gpm
      (infer {:n-grid 30})))
  ([gpm {:keys [n-grid]}]
   (cond
     (view/view? gpm) (infer-alpha gpm n-grid)
     (xcat/xcat? gpm) (infer-alpha gpm n-grid)
     :else (throw (ex-info (str "Concentration hyperparameter inference cannot operate"
                                " on GPM of type: "
                                (type gpm))
                           {:gpm-type (type gpm)})))))
