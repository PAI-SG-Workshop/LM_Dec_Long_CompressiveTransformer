# LM_Dec_Long_CompressiveTransformer

CompressiveTransformer - Long family, decoder-only byte-level language model, standardised in the Statera Working Group Language Model library.

## Status

- Verification level: **Reviewed**
- Failure class: Experiment Incomplete: not run yet

## Provenance

- Source repository: (new) lucidrains/compressive-transformer-pytorch
- Upstream / license: lucidrains/compressive-transformer-pytorch (MIT)

## Differences from the source ([Unified] protocol)

- segments of 128, memory 128 and compressed memory 32 (ratio 4) carried across segments

All models share one protocol: 256-byte vocabulary, dim 512, 6 layers, 8 heads x 64, Adam 2e-4 with 200 warm-up steps, gradient clip 0.5, bf16 autocast (automatic fp32 fallback). Numbers rank structures under one budget on one GPU (RTX 3070 Ti 8 GB); they do not reproduce paper results.

## Results

- No results yet.

## Tests

- check: data counts, causality (no look-ahead, outputs use the current byte), parameters
- scaling: peak GPU memory and time vs length 512-8192
- overfit: memorise 4 x 256 bytes below 0.5 nats
- LM-02: enwik8 next-byte prediction (bits per byte)
- LM-05: long-horizon key-value recall with key-removed control (v3: distinct queries)
- LM-06: memory update - latest value vs stale value after a key is reassigned

## Run

```bash
pip install torch==2.5.1 einops numpy compressive-transformer-pytorch==0.4.0 mogrifier
python run.py check compressive
python run.py lm02 compressive      # enwik8 is downloaded to data/ on first use
```
