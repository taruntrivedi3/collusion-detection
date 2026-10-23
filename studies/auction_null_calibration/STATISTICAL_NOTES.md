# Statistical interpretation

## A controlled product-of-paths null

Let the frozen donor bank contain 50 observed paths. Each path contains both action views for all 29 usable rounds. For artificial agent i, independently draw J_i uniformly from 1,...,50 and set its path X_i equal to donor J_i. Conditional on the bank,

```
P(X_1 in A_1, ..., X_5 in A_5) = product_i P(X_i in A_i).
```

Thus entire random paths are independent. Their source agents' own market behavior need not have been noncolluding for this conditional product construction to be well defined. It is an empirical no-interaction benchmark, not five agents trading in a newly generated market.

Independent draws permit the same donor identity to be selected twice. With 50 donors and five agents, this occurs with probability `1 - (50*49*48*47*46)/50^5`, about 18.6%. The same-pool reference includes these coincidences. Conditioning afterward on distinct identities changes the sampling distribution, so that diagnostic is reported separately rather than used to redefine the primary null.

## Independence of paths does not imply zero pooled-time MI

Consider two independent agents that both tend to quote high early and low late. If time is ignored, learning one agent's quote gives information about the time and hence about the other quote. A shared deterministic schedule can produce pooled dependence without interaction between the agents. The same issue remains inside a ten-round block if behavior keeps changing across those ten rounds.

Within-state shuffling assumes the chosen within-state reorderings give a suitable reference. Circular shifts preserve individual cyclic order but can move late-market behavior into early rounds. Neither transformation generally preserves nonstationary action distributions given exact time and market history. High benchmark flag rates therefore show these particular coarse-state surrogates cannot be used as calibrated no-interaction comparisons for these paths. They do not refute a randomization theorem whose exchangeability assumptions hold, and they do not measure actual-market collusion false positives.

## Why the empirical path reference is valid only for its specified distribution

Fix a normalization bank independently of the observed and reference groups. When an observed group and 999 reference groups are independently generated from the same frozen product-of-paths distribution, their scalar maximum statistics are exchangeable. The usual upper-tail rank p-value is then super-uniform; ties can make it conservative. This argument permits arbitrary temporal behavior within each path.

Changing from training-market paths to held-out-market paths changes the distribution. Training on more or fewer distinct donor identities also changes exact-copy probabilities. The original transfer experiment intentionally exposes the learned-reference mismatch, and the matched-size supplement controls the identity-count difference. Neither experiment supplies additional independent real LLM markets: all the synthetic counts are conditional on a small fixed set of original markets.

The source-style score can also be empirically calibrated by treating the entire randomized A/B procedure as a scalar-valued algorithm and evaluating it on independent benchmark groups. Direct calibration of the maximum avoids saturation of its 999-draw surrogate p-values. That calibration is still conditional on the chosen empirical source distribution; it does not solve transfer to real interacting markets.

## Why exact-time conditioning is powerless here

Every actual episode supplies one seller action vector per round. Exact-time strata therefore have one observation each. Plug-in joint and marginal entropies within each stratum are all zero, so estimated conditional MI is zero. This remains true when three agents' paths are perfect copies. The absence of an empirical signal is a consequence of sample size, not a statement about the underlying conditional joint distribution. Multiple independent episodes or a defensible model that shares information across contexts are needed for richer conditioning.

## Claims these results support

Report surrogate sensitivity, failure under a precisely specified empirical no-interaction benchmark, and the limitations of a reference estimated from a small set of source markets. A calibrated same-pool benchmark and a failed transfer are compatible. Actual no-message markets remain comparison controls rather than certified noncolluding ground truth. Price agreements with constant or independently randomized actions can escape dependence-based scans, and significant groups need not have correct membership. A valid collusion claim also needs an economic/behavioral definition beyond statistical dependence.
