# Contributing

## Running the tests

The test suite makes use of [`kaocha-cljs`](https://github.com/lambdaisland/kaocha-cljs#quickstart), which requires that the JavaScript package `ws` be installed.

```
yarn install ws
```

Once `ws` is installed the test suite can be run with [Kaocha](https://github.com/lambdaisland/kaocha):

```
bin/kaocha
```

### Generating test plots

Some of the tests in the test suite write [Vega](https://vega.github.io/vega/) specifications to the filesystem as a side effect. These specifications can be turned into plots that can be used to evaluate inference quality. `make plots` will turn the Vega specifications into `.png` files and write them to `out`.

`make plots` uses [`vega-cli`] to generate charts. `vega-cli` depends upon `node-canvas`. Occasionally some system configurations may run into errors while installing node-canvas. Please consult the node-canvas documentation if you experience installation issues.
