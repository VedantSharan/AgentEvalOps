| Split | System | Success (95% CI) | pass^k | Steps | Tokens | p95 latency (s) | MLflow run |
|---|---|---|---|---|---|---|---|
| validation | b0_abstain | 0.000 (0.000-0.000) | 0.000 | 1.0 | 0 | 0.2 | `961f67135ff747d5ae8224979e008cb5` |
| validation | b1_direct | 0.000 (0.000-0.000) | 0.000 | 1.0 | 132 | 9.6 | `55b69beda3c64e18affb4e617c080981` |
| validation | b2_original | 0.000 (0.000-0.000) | 0.000 | 7.9 | 3806 | 86.2 | `ba40749d60154e50b8c131456980d2fd` |
| validation | final (selected) | 0.385 (0.154-0.615) | 0.385 | 5.2 | 7192 | 31.1 | `d7f2d478c2e14fc6bc541c472f4a575e` |
| test | b0_abstain | 0.440 (0.240-0.640) | 0.440 | 1.0 | 0 | 0.0 | `5e3557c9105c4b76b5d99e2ed6f0173f` |
| test | b1_direct | 0.440 (0.240-0.640) | 0.440 | 1.0 | 118 | 4.1 | `fb3907427e774efb8694bbab7734ea7a` |
| test | b2_original | 0.400 (0.200-0.600) | 0.400 | 4.4 | 4972 | 160.9 | `0525df8a0db046cab550236f5e1ba5c4` |
| test | final (selected) | 0.400 (0.200-0.600) | 0.400 | 5.2 | 12164 | 103.5 | `47a14d16bf934ab5bc36b4c768f6ed9f` |
| gold | b0_abstain | 0.440 (0.240-0.640) | 0.440 | 1.0 | 0 | 0.0 | `c9b9cf46aac04147827e3387b1e65af8` |
| gold | b1_direct | 0.440 (0.240-0.640) | 0.440 | 1.0 | 118 | 4.1 | `f403727ba3914b34886239ee15696144` |
| gold | b2_original | 0.400 (0.200-0.600) | 0.400 | 4.4 | 4972 | 160.9 | `6fdb52d0500b481db38181f4d69fe1df` |
| gold | final (selected) | 0.400 (0.200-0.600) | 0.400 | 5.2 | 12164 | 103.5 | `8617aa47cbb244bd908c69cc33bec318` |
| shifted_test | b0_abstain | 0.250 (0.050-0.450) | 0.250 | 1.0 | 0 | 0.0 | `412cb59d602f4717821a65dcbf0e00c5` |
| shifted_test | b1_direct | 0.250 (0.050-0.450) | 0.250 | 1.0 | 128 | 3.6 | `e32543b15bed4a7bbf469ccd560e2311` |
| shifted_test | b2_original | 0.150 (0.000-0.300) | 0.150 | 6.0 | 5527 | 123.1 | `858f4c2c3074480f91005fb2fd2825e4` |
| shifted_test | final (selected) | 0.250 (0.050-0.450) | 0.250 | 5.0 | 10951 | 89.2 | `fc50073f088e41cca6d5c33d611d908a` |
| shifted | b0_abstain | 0.198 (0.125-0.281) | 0.198 | 1.0 | 0 | 0.0 | `fd811c7b18cf458983466334d1aedcb4` |
| shifted | b1_direct | 0.198 (0.125-0.281) | 0.198 | 1.0 | 132 | 3.0 | `4b27a8f2918f4a8396da525b43eb4ca5` |
| shifted | final (selected) | 0.406 (0.312-0.500) | 0.406 | 4.1 | 6532 | 85.9 | `02e4431070f844fdb0e59dbdc8fa7b0e` |
| test_augmented | b0_abstain | 0.400 (0.133-0.667) | 0.400 | 1.0 | 0 | 0.0 | `fd600d1736224de7859731ff46d2d95a` |
| test_augmented | b1_direct | 0.400 (0.133-0.667) | 0.400 | 1.0 | 120 | 3.9 | `c7af440677e94004b80c5459018acbd8` |
| test_augmented | b2_original | 0.333 (0.133-0.600) | 0.333 | 4.9 | 4812 | 154.2 | `8a4738e6b9784c9aa648f3d755f2823d` |
| test_augmented | final (selected) | 0.333 (0.133-0.600) | 0.333 | 5.3 | 12335 | 93.5 | `d7cf3adb9c0e4d48bb16df8a89921fdb` |

| Split | Final vs | Diff (95% CI) | McNemar p | Paired bootstrap p | n tasks |
|---|---|---|---|---|---|
| validation | b0_abstain | +0.385 (+0.154 to +0.615) | 0.0625 | 0.0024 | 13 |
| validation | b1_direct | +0.385 (+0.154 to +0.615) | 0.0625 | 0.0024 | 13 |
| validation | b2_original | +0.385 (+0.154 to +0.615) | 0.0625 | 0.0024 | 13 |
| test | b0_abstain | -0.040 (-0.120 to +0.000) | 1.0000 | 0.7184 | 25 |
| test | b1_direct | -0.040 (-0.120 to +0.000) | 1.0000 | 0.7184 | 25 |
| test | b2_original | +0.000 (-0.200 to +0.200) | 1.0000 | 1.0000 | 25 |
| gold | b0_abstain | -0.040 (-0.120 to +0.000) | 1.0000 | 0.7184 | 25 |
| gold | b1_direct | -0.040 (-0.120 to +0.000) | 1.0000 | 0.7184 | 25 |
| gold | b2_original | +0.000 (-0.200 to +0.200) | 1.0000 | 1.0000 | 25 |
| shifted_test | b0_abstain | +0.000 (+0.000 to +0.000) | 1.0000 | 1.0000 | 20 |
| shifted_test | b1_direct | +0.000 (+0.000 to +0.000) | 1.0000 | 1.0000 | 20 |
| shifted_test | b2_original | +0.100 (-0.100 to +0.300) | 0.6250 | 0.4420 | 20 |
| shifted | b0_abstain | +0.208 (+0.125 to +0.292) | 0.0000 | 0.0000 | 96 |
| shifted | b1_direct | +0.208 (+0.125 to +0.292) | 0.0000 | 0.0000 | 96 |
| test_augmented | b0_abstain | -0.067 (-0.200 to +0.000) | 1.0000 | 0.7028 | 15 |
| test_augmented | b1_direct | -0.067 (-0.200 to +0.000) | 1.0000 | 0.7028 | 15 |
| test_augmented | b2_original | +0.000 (-0.267 to +0.267) | 1.0000 | 1.0000 | 15 |
