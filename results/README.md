# Reproduction evidence

The frozen implementation was evaluated with Qwen3-1.7B, TP2/SP, BF16, 4096 tokens and OpenWebMath on H100. The long study contains 20 runs of 201 optimizer updates. The first 20 updates in each run are excluded from timing statistics.

long201.json contains the per-update time and loss, check coverage and paired summaries. It contains no model weights, text samples, server addresses or absolute server paths. For each configuration, matching updates in two mirrored runs are averaged. Speedups are the median of the per-update baseline/candidate ratios. They can differ from the ratio of the two reported median times.

| Comparison | Paired speedup |
| --- | ---: |
| All three vs original, single-path block | 1.17811 |
| All three vs original, double-path block | 1.13537 |
| All three vs AG+RS | 1.03580 |
| All three vs AG+AR | 1.03456 |
| All three vs RS+AR | 1.03032 |

The four individual combined/original repeat speedups span 1.11637–1.17177. AG+RS versus the combined path is the weakest repeat comparison. One repeat gives 0.99941, the other 1.07390. Thus the aggregate combined gain is positive, while repeat-by-repeat superiority over every double path is unproven. All runs completed and passed their 28-layer output/gradient checks with no skipped or NaN iterations. The maximum combined/original loss difference is 0.0052131.

The first original run recorded a foreign GPU process during initialization. Its recorded timestamps precede the measured training window. The timed-window check passed. The remaining run monitoring reports are empty.

Historical pure-kernel measurements for the retained implementations are AG 260.22→254.58 µs, RS 155.90→141.33 µs, AR 129.44→119.22 µs. These compare equal output boundaries at fixed resource counts and exclude external clearing and rank barriers from the CUDA event interval.

The package layout and Megatron adapter were simplified after the frozen experiment. package-validation.json records the checks on the cleaned release. Historical measurements are reproduced in long201.json, rather than attributed to an unmeasured long run of the cleaned adapter.

The cleaned package passed three optimizer updates for all eight configurations, with MLP and projection checks in all 28 layers, plus 28 residual/norm checks whenever AR path optimization was enabled. Its standalone kernels passed reference checks and 21 paired timing groups each. These kernel smoke tests used PyTorch 2.14.1+cu130; the training smoke tests used the historical Pai environment with PyTorch 2.4.0+cu121. They are packaging checks and do not establish a new long-run performance result.

One cold-start baseline run stalled and reached the 240-second timeout; its independent retry completed. Intermittent stalls are therefore not claimed to be fully resolved. The launcher now cleans descendant processes across torchrun sessions, and that timeout behavior was verified separately. A mixed-Python-path import failure was also corrected before the final RS and AG+RS validation. Details are retained in package-validation.json.

release-source-sha256.json identifies the final Python, shell and native source files. These hashes match the files used on the validation server. Each retained kernel function also matches the frozen source at the Python AST level.
