from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any
import math
import zlib

import numpy as np
from PIL import Image


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _pil_to_uint8(image: Image.Image, size: int | None = None) -> Image.Image:
    if size is not None and image.size != (size, size):
        return image.resize((size, size), Image.Resampling.LANCZOS)
    return image.convert("RGB")


@lru_cache(maxsize=1)
def _load_blip(model_name: str):
    from transformers import BlipForConditionalGeneration, BlipProcessor

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    model.to(_device())
    model.eval()
    return processor, model


class BlipCaptioner:
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base") -> None:
        self.model_name = model_name

    def caption(self, image: Image.Image, prompt: str | None = None, max_new_tokens: int = 80) -> str:
        import torch

        processor, model = _load_blip(self.model_name)
        image = _pil_to_uint8(image)
        text = prompt if prompt else ""
        inputs = processor(images=image, text=text, return_tensors="pt")
        inputs = {k: v.to(_device()) for k, v in inputs.items()}
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
        return processor.decode(generated[0], skip_special_tokens=True).strip()


@lru_cache(maxsize=1)
def _load_qwen25_vl(model_name: str):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    dtype = torch.float16 if _device() != "cpu" else torch.float32
    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=256 * 28 * 28,
        max_pixels=768 * 28 * 28,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    model.to(_device())
    model.eval()
    return processor, model


class Qwen25VlFactExtractor:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct") -> None:
        self.model_name = model_name

    def analyze_image(self, image_path: str, prompt: str, max_new_tokens: int = 192) -> str:
        import torch

        processor, model = _load_qwen25_vl(self.model_name)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0].strip()


@lru_cache(maxsize=1)
def _load_sd_turbo(model_name: str):
    import torch
    from diffusers import AutoPipelineForText2Image

    dtype = torch.float16 if _device() != "cpu" else torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained(model_name, torch_dtype=dtype)
    pipe.to(_device())
    pipe.set_progress_bar_config(disable=True)
    return pipe


class SdTurboGenerator:
    def __init__(self, model_name: str = "stabilityai/sd-turbo", num_inference_steps: int = 2, guidance_scale: float = 0.0) -> None:
        self.model_name = model_name
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale

    def generate(self, prompt: str, image_size: int, seed: int = 0) -> Image.Image:
        import torch

        pipe = _load_sd_turbo(self.model_name)
        generator = torch.Generator(device=_device()).manual_seed(seed)
        result = pipe(
            prompt=prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            height=image_size,
            width=image_size,
            generator=generator,
        )
        return result.images[0].convert("RGB")


@lru_cache(maxsize=1)
def _load_vae(model_name: str):
    import torch
    from diffusers import AutoencoderKL

    dtype = torch.float16 if _device() != "cpu" else torch.float32
    vae = AutoencoderKL.from_pretrained(model_name, torch_dtype=dtype)
    vae.to(_device())
    vae.eval()
    return vae


def _image_to_tensor(image: Image.Image, image_size: int, dtype=None):
    import torch

    image = _pil_to_uint8(image, size=image_size)
    arr = np.asarray(image).astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    target_dtype = dtype if dtype is not None else torch.float32
    return tensor.to(device=_device(), dtype=target_dtype)


def _tensor_to_image(tensor) -> Image.Image:
    import torch

    tensor = tensor.detach().to("cpu")
    if tensor.ndim == 4:
        tensor = tensor[0]
    tensor = tensor.clamp(-1.0, 1.0)
    arr = ((tensor.permute(1, 2, 0).numpy() + 1.0) * 127.5).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


class SdvVaeAdapter:
    def __init__(self, model_name: str = "stabilityai/sd-vae-ft-mse") -> None:
        self.model_name = model_name

    def encode(self, image: Image.Image, image_size: int):
        import torch

        vae = _load_vae(self.model_name)
        vae_dtype = next(vae.parameters()).dtype
        tensor = _image_to_tensor(image, image_size=image_size, dtype=vae_dtype)
        with torch.no_grad():
            latent_dist = vae.encode(tensor).latent_dist
            latent = latent_dist.mean * float(vae.config.scaling_factor)
        return latent

    def decode(self, latent):
        import torch
        import torch.nn.functional as F

        vae = _load_vae(self.model_name)
        vae_dtype = next(vae.parameters()).dtype
        latent = latent.to(device=_device(), dtype=vae_dtype)
        with torch.no_grad():
            decoded = vae.decode(latent / float(vae.config.scaling_factor)).sample
        return _tensor_to_image(decoded)


@dataclass
class QuantizedLatent:
    quantized: np.ndarray
    mins: np.ndarray
    maxs: np.ndarray
    shape: tuple[int, int, int]


def quantize_latent(latent_tensor, target_hw: tuple[int, int]) -> QuantizedLatent:
    import torch
    import torch.nn.functional as F

    resized = F.interpolate(latent_tensor, size=target_hw, mode="bilinear", align_corners=False)
    arr = resized.detach().to("cpu").numpy()[0].astype(np.float32)
    mins = arr.reshape(arr.shape[0], -1).min(axis=1)
    maxs = arr.reshape(arr.shape[0], -1).max(axis=1)
    scales = np.maximum(maxs - mins, 1e-6)
    normalized = (arr - mins[:, None, None]) / scales[:, None, None]
    quantized = np.clip(np.round(normalized * 255.0), 0, 255).astype(np.uint8)
    return QuantizedLatent(quantized=quantized, mins=mins.astype(np.float16), maxs=maxs.astype(np.float16), shape=quantized.shape)


def serialize_quantized_latent(q: QuantizedLatent) -> bytes:
    import struct

    channels, height, width = q.shape
    header = struct.pack(">BBB", channels, height, width)
    stats = q.mins.tobytes() + q.maxs.tobytes()
    body = q.quantized.tobytes()
    return zlib.compress(header + stats + body, level=9)


def deserialize_quantized_latent(payload: bytes):
    import struct
    import torch

    raw = zlib.decompress(payload)
    channels, height, width = struct.unpack(">BBB", raw[:3])
    stat_bytes = channels * 2
    mins = np.frombuffer(raw[3 : 3 + stat_bytes], dtype=np.float16).astype(np.float32)
    maxs = np.frombuffer(raw[3 + stat_bytes : 3 + 2 * stat_bytes], dtype=np.float16).astype(np.float32)
    body = raw[3 + 2 * stat_bytes :]
    quantized = np.frombuffer(body, dtype=np.uint8).reshape(channels, height, width).astype(np.float32)
    scales = np.maximum(maxs - mins, 1e-6)
    arr = mins[:, None, None] + (quantized / 255.0) * scales[:, None, None]
    tensor = torch.from_numpy(arr).unsqueeze(0).to(_device())
    return tensor


@lru_cache(maxsize=4)
def _load_semantic_model(model_name: str):
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(_device())
    model.eval()
    return processor, model


class SemanticImageSimilarity:
    def __init__(self, model_name: str = "google/siglip-base-patch16-224") -> None:
        self.model_name = model_name

    def image_similarity(self, image_a: Image.Image, image_b: Image.Image) -> float:
        import torch

        processor, model = _load_semantic_model(self.model_name)
        inputs = processor(images=[_pil_to_uint8(image_a), _pil_to_uint8(image_b)], return_tensors="pt")
        inputs = {k: v.to(_device()) for k, v in inputs.items()}
        with torch.no_grad():
            features = _extract_semantic_image_features(model, inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return float((features[0] * features[1]).sum().detach().cpu().item())


def _extract_semantic_image_features(model, inputs):
    import torch

    if hasattr(model, "get_image_features"):
        features = model.get_image_features(**inputs)
        if isinstance(features, torch.Tensor):
            return features
        if hasattr(features, "image_embeds") and isinstance(features.image_embeds, torch.Tensor):
            return features.image_embeds
        if hasattr(features, "pooler_output") and isinstance(features.pooler_output, torch.Tensor):
            return features.pooler_output
        if hasattr(features, "last_hidden_state") and isinstance(features.last_hidden_state, torch.Tensor):
            return features.last_hidden_state[:, 0]
        raise TypeError(f"Unsupported get_image_features() return type: {type(features)!r}")

    outputs = model(**inputs)
    if hasattr(outputs, "image_embeds") and isinstance(outputs.image_embeds, torch.Tensor):
        return outputs.image_embeds
    if hasattr(outputs, "last_hidden_state") and isinstance(outputs.last_hidden_state, torch.Tensor):
        return outputs.last_hidden_state[:, 0]
    raise TypeError(f"Unsupported semantic model output type: {type(outputs)!r}")


@lru_cache(maxsize=1)
def _load_dino(model_name: str):
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(_device())
    model.eval()
    return processor, model


class DinoSimilarity:
    def __init__(self, model_name: str = "facebook/dinov2-base") -> None:
        self.model_name = model_name

    def image_similarity(self, image_a: Image.Image, image_b: Image.Image) -> float:
        import torch

        processor, model = _load_dino(self.model_name)
        inputs = processor(images=[_pil_to_uint8(image_a), _pil_to_uint8(image_b)], return_tensors="pt")
        inputs = {k: v.to(_device()) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            feats = outputs.last_hidden_state[:, 0]
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return float((feats[0] * feats[1]).sum().detach().cpu().item())


@lru_cache(maxsize=1)
def _load_detr(model_name: str):
    from transformers import DetrForObjectDetection, DetrImageProcessor

    processor = DetrImageProcessor.from_pretrained(model_name)
    model = DetrForObjectDetection.from_pretrained(model_name)
    model.to(_device())
    model.eval()
    return processor, model


class DetrObjectDetector:
    def __init__(self, model_name: str = "facebook/detr-resnet-50", threshold: float = 0.7) -> None:
        self.model_name = model_name
        self.threshold = threshold

    def detect(self, image: Image.Image) -> list[dict[str, Any]]:
        import torch

        processor, model = _load_detr(self.model_name)
        image = _pil_to_uint8(image)
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(_device()) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_object_detection(
            outputs,
            threshold=self.threshold,
            target_sizes=[(image.height, image.width)],
        )[0]
        labels = model.config.id2label
        detections: list[dict[str, Any]] = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections.append(
                {
                    "label": labels[int(label)],
                    "score": float(score.detach().cpu().item()),
                    "bbox": [float(x) for x in box.detach().cpu().tolist()],
                }
            )
        return detections
