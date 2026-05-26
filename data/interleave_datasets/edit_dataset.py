# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io
import random
from PIL import Image, ImageFile, PngImagePlugin

from .interleave_t2i_dataset import InterleavedBaseIterableDataset, ParquetStandardIterableDataset
from ..data_utils import pil_img2rgb


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class UnifiedEditIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):

    def __init__(self, enable_vae=True, **kwargs):
        """
        Args:
            enable_vae: Whether to enable VAE processing for input images.
                        Set to False for pure VLM training without image generation.
        """
        super().__init__(**kwargs)
        self.enable_vae = enable_vae

    def parse_row(self, row):
        data = self._init_data()

        def _normalize_list(val):
            if val is None:
                return []
            if hasattr(val, "tolist"):
                val = val.tolist()
            if isinstance(val, (list, tuple)):
                return list(val)
            return [val]

        instrs = _normalize_list(row.get("instruction_list", None))
        outputs = _normalize_list(row.get("output_text_list", None))

        # Backward compatible: support multiple input formats
        # 1. input_image_list/output_image_list: explicit split (multi-input datasets)
        # 2. image_list + num_input_images: unified format with configurable split
        # 3. image_list only: legacy format (first image is input, rest are outputs)
        if "input_image_list" in row:
            input_images = _normalize_list(row.get("input_image_list", None))
            output_images = _normalize_list(row.get("output_image_list", None))
        else:
            images = _normalize_list(row.get("image_list", None))
            # Use num_input_images to determine split, default to 1 for backward compatibility
            num_inputs = row.get("num_input_images", 1)
            if num_inputs is None:
                num_inputs = 1
            input_images = images[:num_inputs]
            output_images = images[num_inputs:]

        # Filter: skip samples with missing or invalid data
        # Use len() instead of bool check because these may be numpy arrays
        if len(instrs) == 0 or len(input_images) == 0 or len(outputs) == 0:
            print(f"Skipping sample: missing instrs/images/outputs")
            return {}

        # Validate and add all input images (conditioning only, no loss)
        # For multi-input datasets (>1 input images), resize to 512 to reduce token count
        # This saves ~12k tokens for 4-input datasets like multi-view counting
        INPUT_MAX_SIZE = 512  # Max size for conditioning images when multiple inputs
        input_imgs = []
        for idx, img_bytes in enumerate(input_images):
            if img_bytes is None or len(img_bytes) == 0:
                print(f"Skipping sample: input image {idx} is None or empty")
                return {}
            try:
                img = pil_img2rgb(Image.open(io.BytesIO(img_bytes)))
                # Resize input images when there are multiple (e.g., multi-view counting)
                if len(input_images) > 1:
                    w, h = img.size
                    if max(w, h) > INPUT_MAX_SIZE:
                        scale = INPUT_MAX_SIZE / max(w, h)
                        new_w, new_h = int(w * scale), int(h * scale)
                        img = img.resize((new_w, new_h), Image.LANCZOS)
            except Exception as e:
                print(f"Skipping sample: failed to open input image {idx}: {e}")
                return {}
            input_imgs.append(img)

        for img in input_imgs:
            # Input images: enable_cfg=False because we must NOT dropout the input image
            # need_vae controlled by self.enable_vae for pure VLM training support
            data = self._add_image(
                data,
                img,
                need_loss=False,
                need_vae=self.enable_vae,
                need_vit=True,
                enable_cfg=False,
            )

        data = self._add_text(data, instrs[0], need_loss=False)

        for idx, out_txt in enumerate(outputs):
            # Output text should NOT be dropped - we need to compute loss on it
            data = self._add_text(data, out_txt, need_loss=True, enable_cfg=False)

            if idx < len(output_images):
                # Validate subsequent images
                if output_images[idx] is None or len(output_images[idx]) == 0:
                    print(f"Skipping sample: image {idx} is None or empty")
                    return {}
                try:
                    img = pil_img2rgb(Image.open(io.BytesIO(output_images[idx])))
                except Exception as e:
                    print(f"Skipping sample: failed to open image {idx}: {e}")
                    return {}
                # Answer images: enable_cfg=True for CFG training
                # is_output=True uses output_transform for controlling generated image size
                data = self._add_image(
                    data,
                    img,
                    need_loss=True,
                    need_vae=True,
                    need_vit=True,
                    enable_cfg=True,
                    is_output=True,
                )

        return data
