from __future__ import annotations

import hashlib
import re

from visual_memory_benchmark.codecs.base import BaseCodec
from visual_memory_benchmark.models.hf_adapters import Qwen25VlFactExtractor, SdTurboGenerator
from visual_memory_benchmark.types import EncodedArtifact, SceneSample

PROMPT_ECHO_PATTERNS = [
    "a concise factual description of",
    "a detailed description of",
    "describe the colors",
    "describe the spatial layout",
    "describe secondary objects",
    "describe the background",
    "describe the scene",
    "the image shows",
]


class RealTextCodec(BaseCodec):
    def __init__(
        self,
        method_name: str,
        image_size: int,
        vlm_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        generator_model_name: str = "stabilityai/sd-turbo",
        max_new_tokens: int = 192,
        num_inference_steps: int = 2,
        guidance_scale: float = 0.0,
        prompt_prefix: str = "A factual scene description:",
    ) -> None:
        super().__init__(method_name=method_name, image_size=image_size)
        self.extractor = Qwen25VlFactExtractor(model_name=vlm_model_name)
        self.generator = SdTurboGenerator(
            model_name=generator_model_name,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        self.max_new_tokens = max_new_tokens
        self.prompt_prefix = prompt_prefix
        self._unit_cache: dict[str, list[dict[str, str | int]]] = {}

    def encode(self, sample: SceneSample, budget_bytes: int) -> EncodedArtifact:
        units = self._unit_cache.get(sample.sample_id)
        if units is None:
            units = self._generate_factual_units(sample)
            self._unit_cache[sample.sample_id] = units

        selected_units = _pack_units_for_budget(units, budget_bytes)
        stored_text = _join_units(selected_units)
        payload = stored_text.encode("utf-8")
        prompt_echo_flag = any(_contains_prompt_echo(unit["text"]) for unit in selected_units)
        return EncodedArtifact(
            method_name=self.method_name,
            payload=payload,
            aux={
                "text": stored_text,
                "text_sha1": hashlib.sha1(payload).hexdigest(),
                "statement_count": len(selected_units),
                "prompt_echo_flag": prompt_echo_flag,
                "selected_units": selected_units,
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

    def _generate_factual_units(self, sample: SceneSample) -> list[dict[str, str | int]]:
        prompt_specs = [
            (
                1,
                (
                    "List 4 short factual bullet points grounded in the image. "
                    "Focus only on the overall scene, the primary subjects, and the main action. "
                    "Use one bullet per fact. Do not include instructions or explanations."
                ),
                128,
            ),
            (
                2,
                (
                    "List up to 8 additional factual bullet points grounded in the image. "
                    "Focus on visible object counts, major colors, attributes, secondary objects, "
                    "and major left/right or foreground/background relationships. "
                    "Use one bullet per fact. Avoid repeating earlier facts."
                ),
                192,
            ),
            (
                3,
                (
                    "List up to 12 additional factual bullet points grounded in the image. "
                    "Focus on finer layout, pose or orientation, background objects, "
                    "relative spatial relations, and other visually verifiable details. "
                    "Use one bullet per fact. Avoid repeating earlier facts or adding guesses."
                ),
                256,
            ),
        ]

        units: list[dict[str, str | int]] = []
        seen: set[str] = set()
        for level, prompt, tokens in prompt_specs:
            generated = self.extractor.analyze_image(
                sample.source_path or "",
                prompt=prompt,
                max_new_tokens=min(tokens, self.max_new_tokens),
            )
            cleaned_units = _extract_grounded_units(generated, prompt=prompt, level=level)
            for unit in cleaned_units:
                key = _normalize_key(unit["text"])
                if key and key not in seen:
                    seen.add(key)
                    units.append(unit)
        return units


def _extract_grounded_units(text: str, prompt: str | None, level: int) -> list[dict[str, str | int]]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if prompt:
        prompt_norm = prompt.strip().lower()
        cleaned_norm = cleaned.lower()
        if cleaned_norm.startswith(prompt_norm):
            cleaned = cleaned[len(prompt) :].strip(" ,.:;-")

    segments = re.split(r"\n+|(?:^|\s)[-*•]\s+|\s+\d+\.\s+|[.;]\s+", cleaned)
    units: list[dict[str, str | int]] = []
    for segment in segments:
        fact = _sanitize_fact(segment)
        if fact:
            units.append({"text": fact, "level": level})
    return units


def _sanitize_fact(text: str) -> str:
    fact = re.sub(r"\s+", " ", text).strip(" ,.;:-")
    if len(fact) < 8:
        return ""
    lowered = fact.lower()
    if _contains_prompt_echo(lowered):
        return ""
    if lowered in {
        "the picture",
        "the image",
        "the scene",
        "a photo",
        "a picture",
    }:
        return ""
    if re.fullmatch(r"(stuffed[, ]*){3,}.*", lowered):
        return ""
    if lowered.startswith("there is no") or lowered.startswith("not visible"):
        return ""
    return fact


def _pack_units_for_budget(units: list[dict[str, str | int]], budget_bytes: int) -> list[dict[str, str | int]]:
    if budget_bytes <= 0:
        return []
    selected: list[dict[str, str | int]] = []
    for unit in units:
        candidate = _join_units(selected + [unit]).encode("utf-8")
        if len(candidate) <= budget_bytes:
            selected.append(unit)
        else:
            if not selected:
                truncated = _truncate_text_preserving_words(str(unit["text"]), budget_bytes)
                if truncated:
                    return [{"text": truncated, "level": unit["level"]}]
            break
    return selected


def _join_units(units: list[dict[str, str | int]]) -> str:
    return ". ".join(str(unit["text"]).rstrip(". ") for unit in units if str(unit["text"]).strip())


def _contains_prompt_echo(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in PROMPT_ECHO_PATTERNS)


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
