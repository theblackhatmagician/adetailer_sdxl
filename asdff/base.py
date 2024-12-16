from __future__ import annotations

import inspect
from typing import Any, Callable, Iterable, List, Mapping, Optional

from diffusers.utils import logging
from PIL import Image

from functools import cached_property

from diffusers import (
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionPipeline,
    StableDiffusionInpaintPipeline,
)

from asdff.utils import (
    ADOutput,
    bbox_padding,
    composite,
    mask_dilate,
    mask_gaussian_blur,
)
from asdff.yolo import yolo_detector

logger = logging.get_logger("diffusers")


DetectorType = Callable[[Image.Image], Optional[List[Image.Image]]]


def ordinal(n: int) -> str:
    d = {1: "st", 2: "nd", 3: "rd"}
    return str(n) + ("th" if 11 <= n % 100 <= 13 else d.get(n % 10, "th"))


class AdPipelineBase:
    def __init__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], (StableDiffusionXLPipeline, StableDiffusionPipeline)):
            # If a pipeline is passed directly
            self.pipe = args[0]
            for attr in dir(self.pipe):
                if not attr.startswith('__'):
                    setattr(self, attr, getattr(self.pipe, attr))
        else:
            # If components are passed as kwargs
            self.pipe = None
            # Map component names to attributes
            component_map = {
                'vae': 'vae',
                'text_encoder': 'text_encoder',
                'text_encoder_2': 'text_encoder_2',
                'tokenizer': 'tokenizer',
                'tokenizer_2': 'tokenizer_2',
                'unet': 'unet',
                'scheduler': 'scheduler',
                'safety_checker': 'safety_checker',
                'feature_extractor': 'feature_extractor',
                # Add mappings for component dictionary keys
                'vae_decoder': 'vae',
                'vae_encoder': 'vae',
                'text_encoder_one': 'text_encoder',
                'text_encoder_two': 'text_encoder_2',
                'tokenizer_one': 'tokenizer',
                'tokenizer_two': 'tokenizer_2'
            }
            
            for key, value in kwargs.items():
                if key in component_map:
                    setattr(self, component_map[key], value)

    @cached_property
    def inpaint_pipeline(self):
        if self.pipe:
            is_xl = isinstance(self.pipe, StableDiffusionXLPipeline)
        else:
            # Detect if XL based on presence of text_encoder_2
            is_xl = hasattr(self, 'text_encoder_2')

        if is_xl:
            print("Loading StableDiffusionXLInpaintPipeline")
            return StableDiffusionXLInpaintPipeline(
                vae=self.vae,
                text_encoder=self.text_encoder,
                text_encoder_2=self.text_encoder_2,
                tokenizer=self.tokenizer,   
                tokenizer_2=self.tokenizer_2,
                unet=self.unet,
                scheduler=self.scheduler,
                feature_extractor=self.feature_extractor,
            )
        else:
            print("Loading StableDiffusionInpaintPipeline")
            return StableDiffusionInpaintPipeline(
                vae=self.vae,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                unet=self.unet,
                scheduler=self.scheduler,
                safety_checker=self.safety_checker if hasattr(self, 'safety_checker') else None,
                feature_extractor=self.feature_extractor if hasattr(self, 'feature_extractor') else None,
            )

    def __call__(  # noqa: C901
        self,
        common: Mapping[str, Any] | None = None,
        inpaint_only: Mapping[str, Any] | None = None,
        images: Image.Image | Iterable[Image.Image] | None = None,
        detectors: DetectorType | Iterable[DetectorType] | None = None,
        mask_dilation: int = 4,
        mask_blur: int = 4,
        mask_padding: int = 32,

        model_path: str = None
    ):
        if common is None:
            common = {}
        if inpaint_only is None:
            inpaint_only = {}
        if "strength" not in inpaint_only:
            inpaint_only = {**inpaint_only, "strength": 0.4}

        if detectors is None:
                detectors = [self.default_detector]
        elif not isinstance(detectors, Iterable):
            detectors = [detectors]

        if images is None:
            print("No Generated image found")
        else:
            txt2img_images = [images] if not isinstance(images, Iterable) else images
            print("Inpainting...")

        init_images = []
        final_images = []

        for i, init_image in enumerate(txt2img_images):
            init_images.append(init_image.copy())
            final_image = None

            for j, detector in enumerate(detectors):
                if model_path:
                    masks = detector(init_image, model_path = model_path)
                else:
                    masks = detector(init_image)
                
                if masks is None:
                    logger.info(
                        f"No object detected on {ordinal(i + 1)} image with {ordinal(j + 1)} detector."
                    )
                    continue

                for k, mask in enumerate(masks):
                    mask = mask.convert("L")
                    mask = mask_dilate(mask, mask_dilation)
                    bbox = mask.getbbox()
                    if bbox is None:
                        logger.info(f"No object in {ordinal(k + 1)} mask.")
                        continue
                    mask = mask_gaussian_blur(mask, mask_blur)
                    bbox_padded = bbox_padding(bbox, init_image.size, mask_padding)
                    print("padded dim:",bbox_padded)
                    inpaint_output = self.process_inpainting(
                        common,
                        inpaint_only,
                        init_image,
                        mask,
                        bbox_padded,
                    )
                    inpaint_image = inpaint_output[0][0]
                    print("generated inpaint dim:",inpaint_image.size) ## remove
                    final_image = composite(
                        init_image,
                        mask,
                        inpaint_image,
                        bbox_padded,
                    )
                    init_image = final_image

            if final_image is not None:
                final_images.append(final_image)

        return ADOutput(images=final_images, init_images=init_images)

    @property
    def default_detector(self) -> Callable[..., list[Image.Image] | None]:
        return yolo_detector

    def _get_inpaint_args(
        self, common: Mapping[str, Any], inpaint_only: Mapping[str, Any]
    ):
        common = dict(common)
        sig = inspect.signature(self.inpaint_pipeline)
        if (
            "control_image" in sig.parameters
            and "control_image" not in common
            and "image" in common
        ):
            common["control_image"] = common.pop("image")
        return {
            **common,
            **inpaint_only,
            "num_images_per_prompt": 1,
            "output_type": "pil",
        }


    def process_inpainting(
        self,
        common: Mapping[str, Any],
        inpaint_only: Mapping[str, Any],
        init_image: Image.Image,
        mask: Image.Image,
        bbox_padded: tuple[int, int, int, int],
    ):
        crop_image = init_image.crop(bbox_padded)
        crop_mask = mask.crop(bbox_padded)
        inpaint_args = self._get_inpaint_args(common, inpaint_only)
        inpaint_args["image"] = crop_image
        inpaint_args["mask_image"] = crop_mask

        if "control_image" in inpaint_args:
            inpaint_args["control_image"] = inpaint_args["control_image"].resize(
                crop_image.size
            )
        return self.inpaint_pipeline(**inpaint_args)
