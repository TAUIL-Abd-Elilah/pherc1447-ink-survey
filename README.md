# PHerc1447 ink survey: a control-calibrated negative, and how to tell one from a broken pipeline

We ran the 18 August First Letters workflow end to end on **PHerc. 1447**, a prize-eligible
volume, using the released [`scrollprize/ink_9um`](https://huggingface.co/scrollprize/ink_9um)
model against the four surface volumes the Open Data bucket already publishes for it.

**Result: no legible ink in those four segments.** ~42 cm² of real surface, 956 sliding 4 cm²
windows per direction, both directions run, measured against a 345-window known-ink control run through the
identical pipeline on the same day.

That negative is the small half of this repo. The useful half is the machinery that makes it
believable, because a "no ink" image and a broken pipeline look exactly alike, and three
separate mechanisms will hand you the first when you actually have the second.

Everything here is MIT. Every number is reproducible from public data with the scripts in
`scripts/`. Nothing was downloaded that is not public, and no discovery is claimed.

---

## 1. The headline number

| | best 4 cm² window | vs control floor (0.1033) | vs control median (0.2382) |
|---|---:|---|---:|
| **control** — PHerc0139 w043, known ink | 0.3332 | — | — |
| PHerc1447 seg1 | 0.1048 | above | 44% |
| PHerc1447 seg2 | 0.0988 | below | 41% |
| PHerc1447 seg3 | 0.1094 | above | 46% |
| PHerc1447 seg4 | **0.1407** | above | **59%** |

Scores are the mean of the label-smoothing-rescaled prediction over a 2 cm × 2 cm window, the
exact area the First Letters prize specifies, swept at a 1/8-side stride.

Several PHerc1447 windows clear the control's *weakest* window — a low bar, since that window
is most likely a blank margin. **None reaches 60% of the control's median, and not one shows
row structure or a letterform.** The strongest region on the scroll, seg4 reverse at
y=4628 x=1157, is granular texture (`figures/05`).

## 2. Why you should believe the negative

Four confounds were closed *before* the verdict, not after a disappointing picture.

### 2.1 The pipeline was validated against a published prediction

The bucket publishes its own ink prediction for the control segment, from a 2.399 µm per-scroll
model. It is the same canvas at a 3.89× scale ratio (matching 9.362/2.399 = 3.90), so it can be
resampled onto our grid and compared directly. Over 88.2% valid overlap:

- **Pearson r = 0.3829**
- **2.73× enrichment** — our output averages 0.5076 inside the reference's top-decile ink and
  0.1858 outside it

So the invocation, checkpoint, layer window and normalization are all doing their job. This is
the step that converts "we saw nothing" into evidence.

### 2.2 CT support was verified before any GPU time

A rendered surface sitting on zero CT returns a **constant** output. That looks exactly like
"no ink" and means the opposite. This is not hypothetical — see
[ScrollPrize/villa#1254](https://github.com/ScrollPrize/villa/issues/1254), where up to 92% of
predicted sheet in sampled blocks sat on identically-zero CT, and a region reporting 46.9%
surface with 100% empty CT returned a constant map.

`scripts/gate.py` checks every input first (`gate.json`):

| segment | shape | nonzero (mid layer) | mean | std | p1–p99 |
|---|---|---:|---:|---:|---|
| seg1 `20250702235910-auto_grown_20250702235910292` | 31×2980×3240 | 52.5% | 106.31 | 30.13 | 34–162 |
| seg2 `20250703025628-auto_grown_20250703025628283` | 31×4100×4260 | 48.8% | 116.46 | 25.69 | 45–164 |
| seg3 `20250703034159-auto_grown_20250703034159599` | 31×3620×5220 | 50.5% | 105.90 | 30.30 | 34–163 |
| seg4 `20251105093211-z_dbg_gen_00320` | 31×13640×8220 | 8.5% | 100.91 | 29.08 | 35–159 |

All four carry real CT. Run this before you believe any negative.

### 2.3 Label smoothing was undone before thresholding

`ink_9um` trains with BCE label smoothing 0.5, so its most confident **no-ink** output sits near
**0.25, not 0**. Threshold or display raw and a live scroll reads as uniformly grey and dead.
Everything here scores on `(p − 0.25) / 0.5` and keeps raw TIFFs raw. It is in the model card,
and it is very easy to miss.

### 2.4 The scale mismatch was measured, not assumed

`ink_9um` was trained at ~9.362–9.6 µm isotropic. PHerc1447 renders are **8.64 µm** — about 8%
finer. If that alone suppressed the model, the whole negative would be an artifact. So we
resampled seg2's XY to 9.362 µm (`scripts/rescale_zarr.py`, bilinear in XY only — depth is the
model's channel axis and must not be touched) and re-ran:

| seg2 | mean (covered) | >0.50 | >0.75 |
|---|---:|---:|---:|
| native 8.64 µm | 0.1055 | 1.60% | 0.44% |
| resampled 9.362 µm | 0.1163 | 1.90% | 0.49% |

**Real but small: +10% on mean, +19% on the >0.5 fraction.** The control sits at 14.8% above
0.5, still ~7.8× higher after the correction. Scale does not explain the result.

### 2.5 One region is not a baseline

A single quiet window proves nothing. On PHerc. Paris 4 — a scroll that certainly has ink — we
previously measured a **0.000%–14.72%** ink rate across 16 CT-selected, ink-blind regions. Its
internal variation exceeds the gap to most scrolls it gets compared against.

So the control here is not a number, it is a **distribution**: 43.56 cm² giving 345 windows,
range 0.1033–0.3332, median 0.2382. PHerc1447's windows are placed inside that distribution
rather than against a single figure.

## 3. The finding we did not expect

**`ink_9um` localises the ink field but does not resolve letters.**

On the control — a scroll in the model's own training set, described by the team as having the
strongest ink signal of those tested — the 9 µm output is dense and low-contrast. Horizontal
row banding *is* visible at 3 cm scale (`figures/02`), so it genuinely tracks the text field,
which is what the 2.73× enrichment measures. But individual glyphs do not resolve. The
published 2.4 µm per-scroll model on the **same physical region** is markedly sparser and
higher-contrast, with resolvable strokes (`figures/06`, left = ours at 9 µm, right = published
at 2.4 µm).

Two consequences:

1. **Scale matters when you judge legibility.** We first looked at a 1 cm² crop and concluded
   the model produced nothing but blobs. That was wrong — row structure is a multi-centimetre
   property and does not exist at 1 cm² in *either* model's output. Judge legibility at ≥3 cm.
2. **"Run the generic checkpoint and look for letters" is not a path to First Letters.** It
   does not produce letters even where ink certainly exists. This matches the team's own note
   that *"per-scroll models still work better"*, and it is why Stage 5 of the workflow post is
   an iterative per-scroll fine-tuning loop rather than a single inference call. A newcomer who
   skips the control will conclude "no ink" on every scroll they try.

## 4. Two defects found on the way

### 4.1 One transient remote read aborts the entire inference

`vesuvius.ink_detection.inference.infer` streaming from the bucket died twice on seg2 with

```
aiohttp.client_exceptions.ClientPayloadError: Response payload is not completed:
<ContentLengthError: 400, 'Not enough data to satisfy content length header
(received 173631 of 507920 bytes)'>. SSLError(1, '[SSL: RECORD_LAYER_FAILURE]')
```

exit 1, no output, minutes of work lost each time. The **second** failure truncated at a
different offset (299,585 of the same 507,920 bytes), so it is not a corrupt object. Confirmed:

- **12 of 12 chunks fetched cleanly over plain sequential HTTPS**, 4,663,415 bytes, zero failures;
- mirrored locally with retries (`scripts/fetch_zarr.py`), the same segment **inferred in about
  a minute**.

Same bytes, same model, same machine — only the read path differs. villa already carries the
remedy for this exact class in
[#1244](https://github.com/ScrollPrize/villa/pull/1244), *"Retry transient remote reads instead
of aborting the whole run."* The ink inference path has no equivalent. On a streaming workload
this size that is a wall, not a nuisance.

*(Aside for anyone debugging these stores: `0/0.0.0` returning 404 is normal. They are sparse;
only populated chunks exist. It is not data loss.)*

### 4.2 Overlap-blend seams are measurable on the stride lattice

With patch 128, overlap 0.5 and Hann blending, predictions carry faint square structure on the
64 px stride grid. Measured on seg1 forward over CT-supported pixels only:

- mean |horizontal gradient| **1.2697** where `x % 64 == 63` vs **1.0536** elsewhere — **1.205×**
- vertically **1.1322** vs **0.9840** — **1.151×**

A ~15–20% gradient excess sitting exactly on the seam lattice. Modest, but it manufactures
edges, and edges are what a human hunting letterforms latches onto.

## 5. What this does and does not license

It retires **four segments, not PHerc1447.** The team's own claim is ink from almost every
scroll with *extensive segmentation*, and four segments is not extensive segmentation.
PHerc1447 publishes **fifteen**; eleven more carry meshes and need only `vc_render_tifxyz`.
That is the honest next step, and it is cheaper than a fresh spiral fit elsewhere.

Nothing here says PHerc1447 has no ink. It says these four surfaces do not show it at 9 µm with
the generic checkpoint, and it gives you the machinery to check whether your own negative means
anything.

## 6. Reproducing

```bash
# 1. gate the inputs before spending GPU time
python scripts/gate.py

# 2. mirror a surface volume with retries (works around 4.1)
python scripts/fetch_zarr.py \
  "PHerc1447/segments/<SEG>/surface-volumes/8.64um-1.2m-116keV-volume-20250521151220.zarr/" \
  ./local/seg.zarr

# 3. inference (villa main; --no-compile because torch.compile is unreliable on Windows)
python -m vesuvius.ink_detection.inference.infer ./local/seg.zarr \
  ink_9um/hybrid_3d2d-seed42/step-075000.pth pred.tif \
  --overlap 0.5 --blend-mode hann --batch-size 4 --direction both --no-compile

# 4. score, with the label-smoothing rescale and 4 cm2 window sweep
python scripts/analyze.py pred.tif PHerc1447
python scripts/summarize.py          # -> results.json
```

Environment: villa `main` @ `5479453a7`, torch 2.13.0+cu126, one RTX 3090,
`scrollprize/ink_9um` seed42 `step-075000.pth`. Per-run numbers in `results.json`,
CT gate in `gate.json`.

## 7. Scope and disclosure

Negative renders from prize-eligible scrolls are publishable — confirmed by a maintainer in
`#general` on 2026-08-27. **No discovery is claimed and none was found.** `ink_9um` was trained
on PHerc. 0139, Scroll 1667, PHerc. Paris 4 and PHerc. 0814, none of which are prize-eligible,
so running it on PHerc1447 involves no training/inference overlap.

This work was done with AI assistance (claude), directed and reviewed by the author.

— TAUIL Abd Elilah
