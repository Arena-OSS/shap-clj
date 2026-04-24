# shap-clj

A Clojure implementation of [SHAP](https://github.com/shap/shap) — Shapley-value
feature attributions for arbitrary model-agnostic predictors. No native
dependencies in the core; optional adapters (Tablecloth, Deep Diamond,
Neanderthal) live behind aliases.

## Layout

```
src/shap_clj/
  shap.clj           Public entry point: `explain`, `background`, `*-explainer`
  model.clj          Model protocol + FnModel/RowFnModel wrappers
  masker.clj         IndependentMasker — (x, coalition) → batch of synthetic rows
  coerce.clj         2-D double-array ↔ seq-of-seqs / Tablecloth / 1-D row
  explanation.clj    The Explanation record + efficiency-residuals, rankings
  shapley.clj        Kernel weights, coalition enumeration + weighted sampling
  linalg.clj         Tiny WLS solver used by KernelSHAP
  plot.clj           Plotly-shaped maps for bar / waterfall / beeswarm / force
  explainer/
    exact.clj        O(2^d) full-enumeration Shapley — the correctness oracle
    kernel.clj       KernelSHAP (Lundberg–Lee 2017), efficiency by construction
    permutation.clj  Paired-permutation sampling estimator
    linear.clj       Closed-form φ = w·(x − μ); infers w by finite differences
  adapter/
    deep_diamond.clj Optional — wraps an Uncomplicate Deep Diamond net

test/shap_clj/       Unit + axiom + cross-explainer + integration tests
examples/shap_clj/   Runnable demos (see "End-to-end example" below)
```

## Public API

A model is anything `shap-clj.model/as-model` can wrap — most commonly a
`(fn [X] predictions)` that takes a 2-D double array and returns a 1-D vector
of numbers. Then:

```clojure
(require '[shap-clj.shap :as shap])

(shap/explain model X {:background bg :method :exact})
(shap/explain model X {:background bg :method :kernel :n-samples 4000})
(shap/explain model X {:background bg :method :permutation :n-permutations 200})
(shap/explain model X {:background bg :method :linear})
```

`explain` returns an `Explanation` record with `:values` (n × d double-array),
`:base-values` (length n), `:data`, `:feature-names`, `:prediction`, `:method`,
`:extra`. Use `shap/efficiency-residuals` to assert Σφ + base = prediction.

### Choosing an explainer

| Method | Cost | When to use |
|--------|------|-------------|
| `:exact` | `O(2^d · B)` | `d ≲ 12`; the correctness oracle for the others |
| `:kernel` | `O(n-samples · B)` | General-purpose model-agnostic; efficiency is enforced by construction (one variable substituted out via the constraint), so residual is ~machine epsilon at any sample size |
| `:permutation` | `O(n-permutations · d · B)` | Cheap per iteration, converges slower than kernel on high-d |
| `:linear` | `O(d · B)` | The model is (nearly) affine. Infers `w`/`b` by finite differences unless you pass `:coefs`/`:intercept` |

`B` is the background row count.

## Aliases (`deps.edn`)

- `:test` — runs the full test suite via `cognitect-labs/test-runner`. Also has `examples/` on its path so example tests can require the demo namespace.
- `:example` — `clj -M:example` runs the tip-predictor demo (see below).
- `:tablecloth` — adds `scicloj/tablecloth`; enables dataset coercion in `shap-clj.coerce`.
- `:dd` — adds `uncomplicate/deep-diamond`; enables the Deep Diamond adapter. Needs the JVM opens flags (already wired in).
- `:neanderthal` — adds Neanderthal for optional faster linear algebra.
- `:all` — all of the above.

## Common commands

```bash
clojure -X:test                                               # all tests
clojure -X:test :nses '[shap-clj.examples.tip-predictor-test]'  # one ns
clojure -M:example                                            # demo output
```

## End-to-end example: the tip predictor

`examples/shap_clj/examples/tip_predictor.clj` defines a tiny 2-layer ReLU
network that predicts a dinner tip from `[bill, party-size, is-weekend]`:

```
h1 = ReLU(0.15·bill + 0.00·party-size + 1.00·is-weekend)
h2 = ReLU(0.10·bill − 0.30·party-size + 0.00·is-weekend)
y  = 1.0·h1 + 0.5·h2 + 0.5
```

All weights are hand-chosen rather than trained. The point is a model that's:

- small enough that `:exact` (O(2³ · 12) = 96 model calls per row) is instant,
- non-linear through ReLU, with cross-feature interactions via the hidden layer,
- linearisable — with `b1 = 0` the model collapses to `w = W2·W1 = [0.20, −0.15, 1.00]`, `b = 0.5`. This linear closure admits the closed-form Shapley `φ_i = w_i (x_i − μ_i)` which the tests pin down to 1e-9.

The `training-data` (12 rows of toy dinner checks) doubles as the SHAP
background. `sample-orders` is the four canonical inputs used for validation:

| Order | Input | Prediction | Notes |
|-------|-------|-----------:|-------|
| 0 | `[50  2 1]` | $11.20 | both hidden units active |
| 1 | `[25  1 0]` |  $5.35 | both hidden units active |
| 2 | `[100 5 1]` | $20.75 | both hidden units active (large) |
| 3 | `[8   4 0]` |  $1.70 | **ReLU clips h2** — true non-linearity |

On rows 0–2 the net equals its linearisation, so `:exact`, `:kernel`,
`:permutation`, `:linear` all agree to four decimals. On row 3 `h2` clips and
the linear explainer's approximation error surfaces (`bill` φ = −7.40 vs the
true −7.27). That's the whole reason the demo prints all four methods side by
side.

### What the test suite pins down

`test/shap_clj/examples/tip_predictor_test.clj` — 25 tests, 83 assertions,
grouped into seven properties:

1. **Model sanity.** Hand-computed predictions (`[11.20 5.35 20.75 1.70]`) match `predict` to 1e-12, both from seq-of-seqs and 2-D-double-array inputs; `predict-row` is consistent with batched `predict`.
2. **Efficiency** `Σφ + base = prediction`. `:exact` to 1e-9, `:kernel` to 1e-8 (enforced by construction), `:permutation` to 5e-3. The linear explainer's efficiency is checked on the linear-head variant instead, where it's exact.
3. **Cross-explainer agreement.** Kernel matches exact to 1e-6 with 5k samples; permutation matches exact to 5e-3 with 2k permutations.
4. **Closed-form validation on the linear head.** `w = W2·W1` is recomputed from `tip/weights` so the closed-form tests stay honest if the weights change. `φ = w·(x − μ)` is checked directly, and the linear and exact explainers are shown to agree to 1e-9 on the affine variant.
5. **Axiom spot-checks on the ReLU net.** *Dummy* (a zero-weight feature gets φ = 0), *symmetry* (swapping two features with equal-distribution background swaps their φ), and *linearity* (φ of αf + βg = αφ(f) + βφ(g)).
6. **Edge cases.**
   - `d = 1`: φ₀ = f(x) − E[f(X)] (verified on a quadratic).
   - Constant model: all φ ≈ 0 under every explainer.
   - Input at background mean: linear-head φ = 0 exactly.
   - Extreme inputs at 10⁶ scale still preserve efficiency.
   - Single-row background still produces a valid Explanation (baseline = f(that one row), attributions explain the gap to the input prediction).
   - Batched explanation of n rows matches n separate single-row explanations.
7. **Bookkeeping.** `:kernel` and `:permutation` are reproducible with fixed seeds; Explanation shape/metadata is correct; `plot.clj` helpers (bar, waterfall, beeswarm, force-like) render for the tip model.

### Symmetry gotcha (worth remembering)

The Shapley symmetry axiom holds iff the features enter the function
symmetrically **and** the background distribution is symmetric in those
features. For `f(x₀, x₁, x₂) = x₀ + x₁`, a background with
`E[X₀] ≠ E[X₁]` will give different φ values even at `x₀ = x₁`. The
symmetry test uses a bg whose first two columns are a pairwise
permutation of each other to make this clean.

## Conventions

- Primitive double arrays everywhere in hot paths. 2-D data is `^"[[D"` (`Class/forName "[[D"`). Type hints are routine; `*warn-on-reflection*` is on in every namespace.
- Seed all randomness. Tests and demos use fixed seeds so outputs are bit-reproducible.
- Backgrounds are small and deterministic (`shap-clj.fixtures/det-background`). No float noise in the test fixtures.
- The efficiency residual (`shap/efficiency-residuals`) is the single strongest correctness signal; new explainers should be tested against it first.
