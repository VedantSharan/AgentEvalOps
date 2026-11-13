| System | Success (95% CI) | Steps | Tokens | p50 / p95 latency (s) | Submitted | MLflow run |
|---|---|---|---|---|---|---|
| B0 trivial: always abstain | 0.000 (0.000-0.000) | 1.0 | 0 | 0.0 / 0.2 | 1.00 | `5e5be2c53e1f4316b2645483e6dcd3e0` |
| B1 unadapted: direct answer, no tools | 0.000 (0.000-0.000) | 1.0 | 132 | 2.4 / 9.6 | 1.00 | `3f31dad140b646f0ae67820eeb5be6f3` |
| B2 original A1 function-calling loop | 0.000 (0.000-0.000) | 7.9 | 3806 | 40.3 / 86.2 | 0.54 | `b91e38c12f954e0a9ac47d17cc2b9293` |
| P proposed agent, no exemplars (D0) | 0.385 (0.154-0.692) | 5.0 | 6919 | 23.0 / 35.9 | 1.00 | `80b65b15065046dab4ef5e317eed21e9` |
| P + train_base exemplars (D1) | 0.385 (0.154-0.615) | 4.8 | 8785 | 33.0 / 46.4 | 1.00 | `9c67da84b45846629c74d1f757779a70` |
| P + augmented + active-learning data (D2) | 0.231 (0.000-0.462) | 4.2 | 10144 | 23.1 / 39.2 | 1.00 | `564bf6f5d0854870b11ce9bdd0ebd3d3` |

| Comparison (B vs A) | A | B | Diff (95% CI) | McNemar p | Paired bootstrap p |
|---|---|---|---|---|---|
| proposed vs b2_original | 0.000 | 0.385 | +0.385 (+0.154 to +0.692) | 0.062 | 0.004 |
| proposed vs b1_direct | 0.000 | 0.385 | +0.385 (+0.154 to +0.692) | 0.062 | 0.004 |
| proposed_train_base vs proposed | 0.385 | 0.385 | +0.000 (-0.231 to +0.231) | 1.000 | 1.000 |
| proposed_train_aug_al vs proposed | 0.385 | 0.231 | -0.154 (-0.385 to +0.000) | 0.500 | 0.226 |
| proposed_train_aug_al vs proposed_train_base | 0.385 | 0.231 | -0.154 (-0.385 to +0.000) | 0.500 | 0.207 |
