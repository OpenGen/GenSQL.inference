(ns gensql.inference.gpm.utils
  (:require [clojure.math :as math]
            [gensql.inference.utils :as utils]))

(defn crp-weights
  "Given a GPM and the number of auxiliary sub-GPMs, calculates the associated CRP weights.
  Expects an XCat or View GPM."
  [gpm m]
  (let [latents (:latents gpm)
        alpha (:alpha latents)
        counts (:counts latents)
        z (apply + alpha (vals counts))
        ;; The below check is added to avoid divide-by-zero errors,
        ;; in the event that no new sub-GPMs are added, but an empty existing
        ;; sub-GPM is used in its place.
        m (if (zero? m) (inc m) m)
        altered-counts (reduce-kv (fn [counts' sub-gpm-name cnt]
                                    ;; Set the auxiliary weight from 0 to (alpha / m) / z
                                    ;; which for m auxiliary sub-GPMs, will sum to alpha / z.
                                    (let [cnt' (if (zero? cnt) (/ alpha m) cnt)]
                                      (assoc counts' sub-gpm-name (math/log (/ cnt' z)))))
                                  {}
                                  counts)]
    (utils/log-normalize altered-counts)))
