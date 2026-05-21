# Replicate deployment — extenso-claude/tada-voice

This fork adds `cog.yaml` + `predict.py` so the upstream HumeAI/tada repo can
be self-hosted on Replicate via GitHub auto-build.

## One-time setup

### 1. Model on Replicate (already created)

`extenso-claude/tada-voice` exists at https://replicate.com/extenso-claude/tada-voice
(Private, Nvidia L40S). No further Replicate UI steps needed.

### 2. GitHub Actions secrets (MANUAL STEP)

Go to https://github.com/extenso-claude/tada/settings/secrets/actions and add
**two repository secrets**:

| Secret name | Value |
|---|---|
| `REPLICATE_API_TOKEN` | Your Replicate API token from https://replicate.com/account/api-tokens |
| `HF_TOKEN` | A Hugging Face token (https://huggingface.co/settings/tokens) on an account that has accepted the Llama 3.2 Community License at https://huggingface.co/meta-llama/Llama-3.2-3B |

### 3. Trigger the build

Once the secrets are set, go to https://github.com/extenso-claude/tada/actions
and run the **Push to Replicate** workflow (or push any commit to `main`).

The workflow:
1. Reclaims disk space on the GitHub runner (~14 GB free is tight for our image)
2. Installs Cog
3. Logs into Replicate
4. Runs `cog push r8.im/extenso-claude/tada-voice --secret HF_TOKEN=...`
5. The `cog.yaml` `run` block uses the mounted `HF_TOKEN` to download Llama-3.2-licensed
   TADA weights INTO the image. The runtime container never needs the token.

Build takes ~30-60 minutes (most of it is downloading Llama 3.2 3B + the
tada-codec into the GitHub runner, then pushing the layered image to Replicate).

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
