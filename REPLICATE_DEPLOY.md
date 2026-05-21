# Replicate deployment — extenso-claude/tada-voice

This fork adds `cog.yaml` + `predict.py` so the upstream HumeAI/tada repo can
be self-hosted on Replicate via GitHub auto-build.

## One-time setup (you do this in the Replicate UI)

1. Go to https://replicate.com/create
2. Model name: `tada-voice` (full URL becomes `extenso-claude/tada-voice`)
3. Visibility: **Private** (recommended — the model exposes voice cloning).
4. Hardware: **Nvidia L40S** (48 GB VRAM, ~$0.001125/sec while running).
5. Source code: **Connect a GitHub repository** → `extenso-claude/tada` →
   default branch `main`.
6. Build settings → **Secrets** → add:
   - Name: `HF_TOKEN`
   - Value: your Hugging Face token (must have accepted Meta Llama 3.2
     Community License at https://huggingface.co/meta-llama/Llama-3.2-3B)
7. Click **Create model**. Replicate auto-builds on push.

## First build

After the model is created, push a commit (or click "Rebuild") to trigger the
first build. Builds take ~10-15 minutes (downloading Llama-3.2 + TADA weights
into the image is the slow part).

## Inputs

See `predict.py:Predictor.predict()` for the full input surface. Highlights:

- `text` — what to say
- `prompt_audio` — reference voice clip (10-30 s, mono, 16-48 kHz)
- `prompt_transcript` — required for non-English; optional but recommended for English
- `language` — `en` default; aligner used for forced alignment
- All TADA `InferenceOptions` knobs exposed (text_temperature, acoustic_cfg_scale,
  num_acoustic_candidates, scorer, etc.)
- `seed` — reproducibility

## Hardware notes

- 3B model in bf16 ≈ 9 GB VRAM. L40S (48 GB) has lots of headroom.
- Codec encoder ≈ 2.5 GB. Lazy-loaded per language.
- Cold start ≈ 1-2 min (loading weights from local image disk).
- Predict time scales with text length and `num_acoustic_candidates`.
  Single-candidate, ~10 s of speech ≈ 5-10 s on L40S.

## Updating

To pull upstream changes from HumeAI/tada:

```bash
git remote add upstream https://github.com/HumeAI/tada.git
git fetch upstream
git merge upstream/main
git push
```

Replicate will auto-rebuild on push.
