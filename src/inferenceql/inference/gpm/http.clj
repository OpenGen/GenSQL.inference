(ns inferenceql.inference.gpm.http
  "Implementation of a GPM that forwards requests over HTTP. See
  `inferenceql.inference.gpm/http` for details."
  (:require [clj-http.client :as http]
            [inferenceql.inference.gpm.conditioned :as conditioned]
            [inferenceql.inference.gpm.proto :as proto]
            [jsonista.core :as json]))

(defrecord HTTP [url]
  proto/GPM

  (logpdf [this targets constraints]
    (let [body (json/write-value-as-string
                {:targets targets
                 :constraints constraints})
          response (http/post (str url "/logpdf")
                              {:accept :json
                               :body body
                               :content-type :json})]
      (json/read-value (:body response))))

  (simulate [this targets constraints]
    (let [body (json/write-value-as-string
                {:targets targets
                 :constraints constraints})
          response (http/post (str url "/simulate")
                              {:accept :json
                               :body body
                               :content-type :json})]
      (json/read-value (:body response)
                       (json/object-mapper {:decode-key-fn keyword}))))

  (mutual-information [this target-a target-b constraints n-samples]
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

  (condition [this targets conditions]
    (conditioned/condition this targets conditions)))
