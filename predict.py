"""Cog predictor for HumeAI/tada zero-shot voice cloning TTS.

Exposes the full TADA InferenceOptions surface so Replicate callers can sweep
every generation parameter when dialing in a target voice.

Required Replicate secret env var:
  HF_TOKEN — Hugging Face token with access to the gated Llama-3.2 license.
             Set this in the Replicate model settings ("Secrets" section).

Reference voice clip format:
  prompt_audio  — clean 10-30 s WAV/MP3, mono, 16-48 kHz.
  prompt_transcript — verbatim transcript of prompt_audio (required for forced
                      alignment; the built-in ASR is English-only, so non-English
                      prompts MUST supply the transcript).
"""

from __future__ import annotations

import os
from pathlib import Path as PyPath

import torch
import torchaudio
from cog import BasePredictor, Input, Path


MODEL_REPO = os.environ.get("TADA_MODEL_REPO", "HumeAI/tada-3b-ml")
CODEC_REPO = os.environ.get("TADA_CODEC_REPO", "HumeAI/tada-codec")
OUTPUT_SAMPLE_RATE = 24000


class Predictor(BasePredictor):
    def setup(self) -> None:
        """One-time setup on container start.

        Replicate's setup() has a 615s timeout. Loading the full 7 GB
        TadaForCausalLM weights blows that, so we ONLY load the encoder
        here. The main model is lazy-loaded on first predict() (which has
        a 30-minute timeout — plenty of headroom for the heavy load).

        All weights are baked into the image at build time and read from
        the local HF cache. HF_HUB_OFFLINE=1 ensures no network calls.
        """
        import sys
        print("[setup] starting", flush=True)

        # Force HF to use only local cache — never phone home.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        print("[setup] importing tada modules...", flush=True)
        from tada.modules.encoder import Encoder
        from tada.modules.tada import InferenceOptions
        print("[setup] tada modules imported", flush=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(f"[setup] device={device}", flush=True)

        try:
            print(f"[setup] loading encoder from {CODEC_REPO} (local-only)...", flush=True)
            self.encoder_en = Encoder.from_pretrained(
                CODEC_REPO, subfolder="encoder", local_files_only=True
            ).to(device)
            print("[setup] encoder loaded", flush=True)
            self._encoder_cache: dict[str, object] = {"en": self.encoder_en}
        except Exception as e:
            print(f"[setup] FAILED: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise

        # Main model is lazy-loaded on first predict() to stay under the
        # 615s setup timeout. self.model is None until then.
        self.model = None
        self.InferenceOptions = InferenceOptions
        print("[setup] done (model will lazy-load on first predict)", flush=True)

    def _ensure_model_loaded(self) -> None:
        """Lazy-load the main TADA model on first predict()."""
        if self.model is not None:
            return
        import sys, time
        print(f"[lazy-load] loading model from {MODEL_REPO} in bf16, device_map=auto...", flush=True)
        t0 = time.time()
        try:
            from tada.modules.tada import TadaForCausalLM
            self.model = TadaForCausalLM.from_pretrained(
                MODEL_REPO,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            self.model.eval()
            print(f"[lazy-load] model loaded in {time.time() - t0:.1f}s", flush=True)
        except Exception as e:
            print(f"[lazy-load] FAILED: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise

    # --------------------------------------------------------------- helpers

    def _get_encoder(self, language: str):
        """Lazy-load language-specific encoders on demand (from local cache)."""
        if language not in self._encoder_cache:
            from tada.modules.encoder import Encoder
            self._encoder_cache[language] = Encoder.from_pretrained(
                CODEC_REPO,
                subfolder="encoder",
                language=language,
            ).to(self.device)
        return self._encoder_cache[language]

    # --------------------------------------------------------------- predict

    def predict(
        self,
        text: str = Input(
            description="Text to synthesize in the prompt voice.",
        ),
        prompt_audio: Path = Input(
            description="Reference voice clip (10-30 s clean WAV/MP3, mono, 16-48 kHz).",
        ),
        prompt_transcript: str = Input(
            description=(
                "Verbatim transcript of prompt_audio. Required for non-English; "
                "for English, the built-in ASR can usually handle it but supplying "
                "the transcript is more reliable."
            ),
            default="",
        ),
        language: str = Input(
            default="en",
            choices=["en", "ar", "ch", "de", "es", "fr", "it", "ja", "pl", "pt"],
            description="Aligner language. Non-English prompts MUST supply prompt_transcript.",
        ),
        # ---- Text/LM sampling ----
        text_do_sample: bool = Input(
            default=True,
            description="Enable sampling on the LM stage (False = greedy).",
        ),
        text_temperature: float = Input(
            default=0.6, ge=0.0, le=2.0,
            description="LM-stage sampling temperature. 0.7 for natural reads.",
        ),
        text_top_k: int = Input(
            default=0, ge=0, le=200,
            description="LM-stage top-k (0 = disabled).",
        ),
        text_top_p: float = Input(
            default=0.9, ge=0.0, le=1.0,
            description="LM-stage nucleus sampling.",
        ),
        text_repetition_penalty: float = Input(
            default=1.1, ge=1.0, le=5.0,
            description="LM-stage repetition penalty.",
        ),
        # ---- Acoustic / CFG ----
        acoustic_cfg_scale: float = Input(
            default=1.6, ge=0.0, le=5.0,
            description="Voice-fidelity CFG scale. 1.3-1.5 looser, 1.8-2.5 tighter to reference.",
        ),
        duration_cfg_scale: float = Input(
            default=1.0, ge=0.0, le=5.0,
            description="Duration-prediction CFG. 0.5-0.8 conversational, 1.5+ rhythmic.",
        ),
        cfg_schedule: str = Input(
            default="cosine",
            choices=["constant", "linear", "cosine"],
            description="How CFG scale changes during generation.",
        ),
        # ---- Flow matching ----
        noise_temperature: float = Input(
            default=0.9, ge=0.0, le=2.0,
            description="Initial flow-matching noise. Lower = more deterministic.",
        ),
        num_flow_matching_steps: int = Input(
            default=10, ge=1, le=50,
            description="Flow-matching diffusion steps. 10 default, 20-30 for highest quality.",
        ),
        time_schedule: str = Input(
            default="logsnr",
            choices=["uniform", "cosine", "logsnr"],
            description="Time-stepping schedule for flow matching.",
        ),
        # ---- Candidates / scoring ----
        num_acoustic_candidates: int = Input(
            default=1, ge=1, le=8,
            description="Generate N candidates, pick best via scorer. N>1 multiplies cost+time.",
        ),
        scorer: str = Input(
            default="likelihood",
            choices=["spkr_verification", "likelihood", "duration_median"],
            description="Candidate ranking method when num_acoustic_candidates > 1.",
        ),
        spkr_verification_weight: float = Input(
            default=1.0, ge=0.0, le=5.0,
            description="Weight applied to speaker-verification score.",
        ),
        # ---- Misc ----
        speed_up_factor: float = Input(
            default=0.0,
            description="Speed-up factor for output pacing. 0 = disabled (use 1.05-1.3 for faster delivery).",
        ),
        negative_condition_source: str = Input(
            default="negative_step_output",
            choices=["negative_step_output", "prompt", "zero"],
            description="CFG negative-condition source.",
        ),
        text_only_logit_scale: float = Input(
            default=0.0, ge=0.0, le=2.0,
            description="Mix-in factor for text-only generation logits.",
        ),
        num_extra_steps: int = Input(
            default=0, ge=0, le=200,
            description="Extra autoregressive steps after text ends (speech continuation).",
        ),
        normalize_text: bool = Input(
            default=True,
            description="Apply text normalization before tokenization.",
        ),
        seed: int = Input(
            default=-1,
            description="Random seed (-1 = random). Same seed + same params = reproducible output.",
        ),
    ) -> Path:
        """Generate speech in the prompt voice."""

        # Lazy-load the main model on first prediction (kept out of setup()
        # to stay under Replicate's 615s setup timeout).
        self._ensure_model_loaded()

        if seed is not None and seed >= 0:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Load reference audio.
        audio, sample_rate = torchaudio.load(str(prompt_audio))
        # Mono — TADA's encoder expects single-channel.
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        audio = audio.to(self.device)

        # Encode prompt with the language-appropriate aligner.
        encoder = self._get_encoder(language)
        transcript_arg = prompt_transcript if prompt_transcript else None
        if transcript_arg is None and language != "en":
            raise RuntimeError(
                f"prompt_transcript is required for language={language!r} "
                "(built-in ASR is English-only)."
            )

        if transcript_arg is not None:
            prompt = encoder(audio, text=[transcript_arg], sample_rate=sample_rate)
        else:
            # English fallback — let the built-in ASR transcribe.
            prompt = encoder(audio, sample_rate=sample_rate)

        # Build InferenceOptions.
        opts = self.InferenceOptions(
            text_do_sample=text_do_sample,
            text_temperature=text_temperature,
            text_top_k=text_top_k,
            text_top_p=text_top_p,
            text_repetition_penalty=text_repetition_penalty,
            acoustic_cfg_scale=acoustic_cfg_scale,
            duration_cfg_scale=duration_cfg_scale,
            cfg_schedule=cfg_schedule,  # type: ignore[arg-type]
            noise_temperature=noise_temperature,
            num_flow_matching_steps=num_flow_matching_steps,
            time_schedule=time_schedule,  # type: ignore[arg-type]
            num_acoustic_candidates=num_acoustic_candidates,
            scorer=scorer,  # type: ignore[arg-type]
            spkr_verification_weight=spkr_verification_weight,
            speed_up_factor=speed_up_factor if speed_up_factor > 0 else None,
            negative_condition_source=negative_condition_source,  # type: ignore[arg-type]
            text_only_logit_scale=text_only_logit_scale,
        )

        with torch.no_grad():
            output = self.model.generate(
                prompt=prompt,
                text=text,
                inference_options=opts,
                num_extra_steps=num_extra_steps,
                normalize_text=normalize_text,
            )

        # output.audio is a list[Tensor] (one per batch entry).
        wav = output.audio[0]
        if wav is None:
            raise RuntimeError("Generation produced empty audio. Try a different reference clip or relax sampling params.")
        wav = wav.detach().cpu().float()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        out_path = PyPath("/tmp/output.wav")
        torchaudio.save(str(out_path), wav, OUTPUT_SAMPLE_RATE)
        return Path(str(out_path))
