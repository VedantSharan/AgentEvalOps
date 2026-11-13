| Split | System | Success (95% CI) | pass^k | Steps | Tokens | p95 latency (s) | MLflow run |
|---|---|---|---|---|---|---|---|
| validation | b0_abstain | 0.000 (0.000-0.000) | 0.000 | 1.0 | 0 | 0.2 | `961f67135ff747d5ae8224979e008cb5` |
| validation | b1_direct | 0.000 (0.000-0.000) | 0.000 | 1.0 | 132 | 9.6 | `55b69beda3c64e18affb4e617c080981` |
| validation | b2_original | 0.000 (0.000-0.000) | 0.000 | 7.9 | 3806 | 86.2 | `ba40749d60154e50b8c131456980d2fd` |
| validation | final (retrain_main_c0) | 0.231 (0.000-0.462) | 0.231 | 4.1 | 9260 | 23.9 | `c269642e8f50429281749845f4d92a3c` |
| test | b0_abstain | 0.440 (0.240-0.640) | 0.440 | 1.0 | 0 | 0.0 | `5e3557c9105c4b76b5d99e2ed6f0173f` |
| test | b1_direct | 0.440 (0.240-0.640) | 0.440 | 1.0 | 118 | 4.1 | `fb3907427e774efb8694bbab7734ea7a` |
| test | b2_original | 0.400 (0.200-0.600) | 0.400 | 4.4 | 4972 | 160.9 | `0525df8a0db046cab550236f5e1ba5c4` |
| test | final (retrain_main_c0) | 0.480 (0.280-0.680) | 0.480 | 2.6 | 5520 | 56.2 | `595b97a9f6464bb0b9a4a6a13a44230b` |
| gold | b0_abstain | 0.440 (0.240-0.640) | 0.440 | 1.0 | 0 | 0.0 | `c9b9cf46aac04147827e3387b1e65af8` |
| gold | b1_direct | 0.440 (0.240-0.640) | 0.440 | 1.0 | 118 | 4.1 | `f403727ba3914b34886239ee15696144` |
| gold | b2_original | 0.400 (0.200-0.600) | 0.400 | 4.4 | 4972 | 160.9 | `6fdb52d0500b481db38181f4d69fe1df` |
| gold | final (retrain_main_c0) | 0.480 (0.280-0.680) | 0.480 | 2.6 | 5520 | 56.2 | `13726561a0874463b84e96b8405c0583` |
| shifted_test | b0_abstain | 0.250 (0.050-0.450) | 0.250 | 1.0 | 0 | 0.0 | `412cb59d602f4717821a65dcbf0e00c5` |
| shifted_test | b1_direct | 0.250 (0.050-0.450) | 0.250 | 1.0 | 128 | 3.6 | `e32543b15bed4a7bbf469ccd560e2311` |
| shifted_test | b2_original | 0.150 (0.000-0.300) | 0.150 | 6.0 | 5527 | 123.1 | `858f4c2c3074480f91005fb2fd2825e4` |
| shifted_test | final (retrain_main_c0) | 0.350 (0.150-0.550) | 0.350 | 3.6 | 8740 | 76.2 | `0bc4f3ffa0b4482ca65ddd380c5900a3` |
| shifted | b0_abstain | 0.198 (0.125-0.281) | 0.198 | 1.0 | 0 | 0.0 | `fd811c7b18cf458983466334d1aedcb4` |
| shifted | b1_direct | 0.198 (0.125-0.281) | 0.198 | 1.0 | 132 | 3.0 | `4b27a8f2918f4a8396da525b43eb4ca5` |
| shifted | final (retrain_main_c0) | 0.396 (0.302-0.500) | 0.396 | 3.9 | 8737 | 43.5 | `3a705fd6b3e641e7b09d36318be09692` |
| test_augmented | b0_abstain | 0.400 (0.133-0.667) | 0.400 | 1.0 | 0 | 0.0 | `fd600d1736224de7859731ff46d2d95a` |
| test_augmented | b1_direct | 0.400 (0.133-0.667) | 0.400 | 1.0 | 120 | 3.9 | `c7af440677e94004b80c5459018acbd8` |
| test_augmented | b2_original | 0.333 (0.133-0.600) | 0.333 | 4.9 | 4812 | 154.2 | `8a4738e6b9784c9aa648f3d755f2823d` |
| test_augmented | final (retrain_main_c0) | 0.400 (0.133-0.667) | 0.400 | 2.5 | 5210 | 48.3 | `e2c9a2e740ba4bcdb477dea6ef11962c` |

| Split | Final vs | Diff (95% CI) | McNemar p | Paired bootstrap p | n tasks |
|---|---|---|---|---|---|
| validation | b0_abstain | +0.231 (+0.000 to +0.462) | 0.2500 | 0.0720 | 13 |
| validation | b1_direct | +0.231 (+0.000 to +0.462) | 0.2500 | 0.0720 | 13 |
| validation | b2_original | +0.231 (+0.000 to +0.462) | 0.2500 | 0.0720 | 13 |
| test | b0_abstain | +0.040 (-0.080 to +0.160) | 1.0000 | 0.7856 | 25 |
| test | b1_direct | +0.040 (-0.080 to +0.160) | 1.0000 | 0.7856 | 25 |
| test | b2_original | +0.080 (-0.080 to +0.240) | 0.6250 | 0.4480 | 25 |
| gold | b0_abstain | +0.040 (-0.080 to +0.160) | 1.0000 | 0.7856 | 25 |
| gold | b1_direct | +0.040 (-0.080 to +0.160) | 1.0000 | 0.7856 | 25 |
| gold | b2_original | +0.080 (-0.080 to +0.240) | 0.6250 | 0.4480 | 25 |
| shifted_test | b0_abstain | +0.100 (+0.000 to +0.250) | 0.5000 | 0.2204 | 20 |
| shifted_test | b1_direct | +0.100 (+0.000 to +0.250) | 0.5000 | 0.2204 | 20 |
| shifted_test | b2_original | +0.200 (+0.000 to +0.400) | 0.2188 | 0.1216 | 20 |
| shifted | b0_abstain | +0.198 (+0.115 to +0.281) | 0.0000 | 0.0000 | 96 |
| shifted | b1_direct | +0.198 (+0.115 to +0.281) | 0.0000 | 0.0000 | 96 |
| test_augmented | b0_abstain | +0.000 (-0.200 to +0.200) | 1.0000 | 1.0000 | 15 |
| test_augmented | b1_direct | +0.000 (-0.200 to +0.200) | 1.0000 | 1.0000 | 15 |
| test_augmented | b2_original | +0.067 (-0.133 to +0.267) | 1.0000 | 0.7736 | 15 |
