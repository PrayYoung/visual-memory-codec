from __future__ import annotations

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
        self._caption_cache: dict[str, str] = {}

    def encode(self, sample: SceneSample, budget_bytes: int) -> EncodedArtifact:
        full_caption = self._caption_cache.get(sample.sample_id)
        if full_caption is None:
            prompts = [
                None,
                "a detailed description of",
                "the scene contains",
            ]
            parts = []
            for prompt in prompts:
                text = self.captioner.caption(sample.image, prompt=prompt, max_new_tokens=self.max_new_tokens)
                if text and text not in parts:
                    parts.append(text)
            full_caption = ". ".join(parts)
            self._caption_cache[sample.sample_id] = full_caption

        budgeted = _budget_text(full_caption, budget_bytes)
        return EncodedArtifact(
            method_name=self.method_name,
            payload=budgeted.encode("utf-8"),
            aux={"text": budgeted},
        )

    def decode(self, artifact: EncodedArtifact):
        prompt = artifact.payload.decode("utf-8").strip()
        if not prompt:
            prompt = "an indoor or outdoor scene"
        if self.prompt_prefix:
            prompt = f"{self.prompt_prefix} {prompt}"
        seed = abs(hash(prompt)) % (2**31)
        return self.generator.generate(prompt, image_size=self.image_size, seed=seed)


def _budget_text(text: str, budget_bytes: int) -> str:
    if len(text.encode("utf-8")) <= budget_bytes:
        return text
    sentences = [segment.strip() for segment in text.split(".") if segment.strip()]
    kept: list[str] = []
    for sentence in sentences:
        candidate = ". ".join(kept + [sentence])
        if len(candidate.encode("utf-8")) <= budget_bytes:
            kept.append(sentence)
        else:
            break
    if kept:
        return ". ".join(kept)
    words = text.split()
    out: list[str] = []
    for word in words:
        candidate = " ".join(out + [word])
        if len(candidate.encode("utf-8")) <= budget_bytes:
            out.append(word)
        else:
            break
    return " ".join(out)
