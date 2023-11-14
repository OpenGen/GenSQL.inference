(ns inferenceql.inference.sample
  (:require [gen.dynamic :as dynamic :refer [gen]]
            [gen.distribution.commons-math :refer [categorical student-t]]
            [inferenceql.inference.gpm.proto :as gpm.proto]))

(def generate-row
  (with-meta
    (gen []
      (case (categorical
             [0.28676840905743073 0.2234441546914367 0.002713896615685466 9.046322052284883E-4])
        0 {"Apogee_km" (dynamic/trace! "Apogee_km" student-t 1416.9999999999995 35812.46031878512 2080.997073363445,)
           "Perigee_km" (dynamic/trace! "Perigee_km" student-t 1416.9999999999995 35750.36348694458 1474.9971069850053),
           "Period_minutes" (dynamic/trace! "Period_minutes" student-t 1415.9999999999998 1435.8845841561713 59.048534012483465)}
        1 {"Apogee_km" (dynamic/trace! "Apogee_km" student-t 1346.9999999999995 964.4720690048606 2140.14014572675),
           "Perigee_km" (dynamic/trace! "Perigee_km" student-t 1346.9999999999995 943.0731071864293 1515.818318865372),
           "Period_minutes" (dynamic/trace! "Period_minutes" student-t 1345.9999999999998 104.2744615743889 60.67251222109236)}
        2 {"Apogee_km" (dynamic/trace! "Apogee_km" student-t 1102.9999999999995 120848.5385097515 2717.83417516179),
           "Perigee_km" (dynamic/trace! "Perigee_km" student-t 1102.9999999999995 17157.581830703708 1918.2955113675462),
           "Period_minutes" (dynamic/trace! "Period_minutes" student-t 1101.9999999999998 3431.467343169398 77.27335117514374)}
        3 {"Apogee_km" (dynamic/trace! "Apogee_km" student-t 1100.9999999999995 358532.5391650224 3351.9032307203747),
           "Perigee_km" (dynamic/trace! "Perigee_km" student-t 1100.9999999999995 35385.24470055898 2346.9660843533156),
           "Period_minutes" (dynamic/trace! "Period_minutes" student-t 1099.9999999999998 1434.9169263391718 94.45306975572818)}))
    {`gpm.proto/variables (constantly #{"Apogee_km" "Perigee_km" "Period_minutes"})}))
