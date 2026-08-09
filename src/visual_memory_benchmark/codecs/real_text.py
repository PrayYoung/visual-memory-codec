from __future__ import annotations

import hashlib
import re

from visual_memory_benchmark.codecs.base import BaseCodec
from visual_memory_benchmark.models.hf_adapters import BlipCaptioner, SdTurboGenerator
from visual_memory_benchmark.types import EncodedArtifact, SceneSample


class RealTextCodec(BaseCodec):
    def __init__(
        self,
        method_name: str,
        image_size: int,
        caption_model_name: str = "Salesforce/blip-image-captioning-base",
        generator_model_name: str = "stabilityai/sd-turbo",
        max_new_tokens: int = 96,
        num_inference_steps: int = 2,
        guidance_scale: float = 0.0,
        prompt_prefix: str = "A factual scene description:",
    ) -> None:
        super().__init__(method_name=method_name, image_size=image_size)
        self.captioner = BlipCaptioner(model_name=caption_model_name)
        self.generator = SdTurboGenerator(
            model_name=generator_model_name,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        self.max_new_tokens = max_new_tokens
        self.prompt_prefix = prompt_prefix
        self._segment_cache: dict[str, list[str]] = {}

    def encode(self, sample: SceneSample, budget_bytes: int) -> EncodedArtifact:
        segments = self._segment_cache.get(sample.sample_id)
        if segments is None:
            segments = self._generate_segments(sample)
            self._segment_cache[sample.sample_id] = segments

        budgeted = _pack_segments_for_budget(segments, budget_bytes)
        payload = budgeted.encode("utf-8")
        return EncodedArtifact(
            method_name=self.method_name,
            payload=payload,
            aux={
                "text": budgeted,
                "text_sha1": hashlib.sha1(payload).hexdigest(),
                "segment_count": len([item for item in budgeted.split(". ") if item.strip()]),
            },
        )

    def decode(self, artifact: EncodedArtifact):
        prompt = artifact.payload.decode("utf-8").strip()
        if not prompt:
            prompt = "an indoor or outdoor scene"
        if self.prompt_prefix:
            prompt = f"{self.prompt_prefix} {prompt}"
        seed = abs(hash(prompt)) % (2**31)
        return self.generator.generate(prompt, image_size=self.image_size, seed=seed)

    def _generate_segments(self, sample: SceneSample) -> list[str]:
        prompt_specs = [
            {
                "prompt": None,
                "max_new_tokens": min(self.max_new_tokens, 48),
            },
            {
                "prompt": "a concise factual description of",
                "max_new_tokens": min(self.max_new_tokens, 64),
            },
            {
                "prompt": "a detailed description of the main objects, people, animals, and actions in",
                "max_new_tokens": max(self.max_new_tokens, 96),
            },
            {
                "prompt": "describe the colors, clothing, appearance, and notable attributes in",
                "max_new_tokens": max(self.max_new_tokens, 96),
            },
            {
                "prompt": "describe the spatial layout, positions, foreground, background, and scene context in",
                "max_new_tokens": max(self.max_new_tokens, 112),
            },
            {
                "prompt": "describe secondary objects, background details, surroundings, and any visible relationships in",
                "max_new_tokens": max(self.max_new_tokens, 128),
            },
        ]

        segments: list[str] = []
        seen_keys: set[str] = set()
        for spec in prompt_specs:
            text = self.captioner.caption(
                sample.image,
                prompt=spec["prompt"],
                max_new_tokens=spec["max_new_tokens"],
            )
            for segment in _split_into_segments(text):
                key = _normalize_key(segment)
                if key and key not in seen_keys:
                    seen_keys.add(key)
                    segments.append(segment)
        return segments


def _pack_segments_for_budget(segments: list[str], budget_bytes: int) -> str:
    if budget_bytes <= 0:
        return ""

    selected: list[str] = []
    for segment in segments:
        candidate = _join_segments(selected + [segment])
        if len(candidate.encode("utf-8")) <= budget_bytes:
            selected.append(segment)
        else:
            if not selected:
                return _truncate_text_preserving_words(segment, budget_bytes)
            break
    return _join_segments(selected)


def _join_segments(segments: list[str]) -> str:
    return ". ".join(segment.rstrip(". ") for segment in segments if segment.strip())


def _split_into_segments(text: str) -> list[str]:
    raw_parts = re.split(r"[.;]\s+|\n+", text.strip())
    segments: list[str] = []
    for part in raw_parts:
        cleaned = re.sub(r"\s+", " ", part).strip(" ,.;:-")
        if len(cleaned) >= 8:
            segments.append(cleaned)
    return segments


def _normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def _truncate_text_preserving_words(text: str, budget_bytes: int) -> str:
    words = text.split()
    kept: list[str] = []
    for word in words:
        candidate = " ".join(kept + [word])
        if len(candidate.encode("utf-8")) <= budget_bytes:
            kept.append(word)
        else:
            break
    return " ".join(kept)
