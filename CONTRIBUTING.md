# Contributing

## Running the tests

The Clojure tests can be run with:

```
clojure -X:test:clj-test
```

The ClojureScript tests can be run with:

```
clojure -X:test:cljs-test
```

### Generating test plots

Some of the tests in the test suite write [Vega](https://vega.github.io/vega/) specifications to the filesystem as a side effect. These specifications can be turned into plots that can be used to evaluate inference quality. `make plots` will turn the Vega specifications into `.png` files and write them to `out`.

`make plots` uses [`vega-cli`] to generate charts. `vega-cli` depends upon `node-canvas`. Occasionally some system configurations may run into errors while installing node-canvas. Please consult the node-canvas documentation if you experience installation issues.
