(ns gensql.inference.gpm.http
  "Implementation of a GPM that forwards requests over HTTP. See
  `gensql.inference.gpm/http` for details."
  (:require [clj-http.client :as http]
            [gensql.inference.gpm.conditioned :as conditioned]
            [gensql.inference.gpm.constrained :as constrained]
            [gensql.inference.gpm.proto :as proto]
            [jsonista.core :as json]))

(defrecord HTTP [url]
  proto/GPM

  (logpdf [_ targets constraints]
    (let [body (json/write-value-as-string
                {:targets targets
                 :constraints constraints})
          response (http/post (str url "/logpdf")
                              {:accept :json
                               :body body
                               :content-type :json})]
      (json/read-value (:body response))))

  (simulate [_ targets constraints]
    (let [body (json/write-value-as-string
                {:targets targets
                 :constraints constraints})
          response (http/post (str url "/simulate")
                              {:accept :json
                               :body body
                               :content-type :json})]
      (json/read-value (:body response)
                       (json/object-mapper {:decode-key-fn keyword}))))

  (mutual-information [_ target-a target-b constraints n-samples]
    (let [body (json/write-value-as-string
                {:target-a target-a
                 :target-b target-b
                 :constraints constraints
                 :n-samples n-samples})
          response (http/post (str url "/mutual-information")
                              {:accept :json
                               :body body
                               :content-type :json})]
      (json/read-value (:body response))))

  proto/Condition

  (condition [this conditions]
    (conditioned/condition this conditions))

  proto/Constrain

  (constrain [this targets conditions]
    (constrained/constrain this targets conditions)))
