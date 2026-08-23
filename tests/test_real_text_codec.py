from __future__ import annotations

from unittest import mock

from PIL import Image

from visual_memory_benchmark.codecs.real_text import RealTextCodec
from visual_memory_benchmark.types import SceneSample


class _FakeExtractor:
    def analyze_image(self, image_path: str, prompt: str, max_new_tokens: int = 0) -> str:
        if "Output at most 12 lines" in prompt:
            return "\n".join(
                [
                    "[L1] two people stand on a beach",
                    "[L2] a blue umbrella is near them",
                ]
            )
        return "\n".join(
            [
                "[L1] two people stand on a beach",
                "[L2] a blue umbrella is near them",
                "[L2] small waves reach the sand in front of them",
                "[L3] one person is on the left wearing a dark shirt",
                "[L3] a pale sky fills the upper background",
                "[L3] footprints and darker wet sand appear in the foreground",
            ]
        )


class _FakeGenerator:
    def generate(self, prompt: str, image_size: int, seed: int = 0):
        return Image.new("RGB", (image_size, image_size), color=(128, 128, 128))


def test_real_text_codec_uses_larger_budget_for_richer_extraction():
    sample = SceneSample(
        sample_id="sample-1",
        image=Image.new("RGB", (32, 32), color=(255, 255, 255)),
        objects=None,
        metadata={},
        source_path="/tmp/fake.png",
    )

    with mock.patch("visual_memory_benchmark.codecs.real_text.Qwen25VlFactExtractor", return_value=_FakeExtractor()):
        with mock.patch("visual_memory_benchmark.codecs.real_text.SdTurboGenerator", return_value=_FakeGenerator()):
            codec = RealTextCodec(method_name="text_only_real", image_size=32)

    low = codec.encode(sample, 2048)
    high = codec.encode(sample, 8192)

    assert low.aux["budget_profile"] == "compact"
    assert high.aux["budget_profile"] == "exhaustive"
    assert high.stored_bytes > low.stored_bytes
    assert high.aux["all_statement_count"] > low.aux["all_statement_count"]
