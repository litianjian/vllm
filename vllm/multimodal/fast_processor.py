"""Fast video processor class for LLaVa-Onevision."""

from typing import Dict, List, Optional, Union

import torch
import torchvision
from transformers import AutoImageProcessor
from transformers.image_processing_utils import (BaseImageProcessor,
                                                 BatchFeature, get_size_dict)
from transformers.image_transforms import convert_to_rgb
from transformers.image_utils import (OPENAI_CLIP_MEAN, OPENAI_CLIP_STD,
                                      ChannelDimension, ImageInput, ImageType,
                                      PILImageResampling, VideoInput,
                                      get_image_type,
                                      infer_channel_dimension_format,
                                      is_scaled_image,
                                      is_torchvision_available, is_valid_image,
                                      to_numpy_array, valid_images,
                                      validate_preprocess_arguments)
from transformers.utils import TensorType, is_vision_available, logging

logger = logging.get_logger(__name__)


def is_torchvision_v2_available():
    # torchvision.__version__ will return a string like '0.8.1+cu101'
    # you can split and compare the major version
    major_version = int(torchvision.__version__.split('.')[0])
    return major_version >= 2


if is_vision_available():
    from PIL import Image

if is_torchvision_available():

    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import functional as F


def make_batched_videos(videos):
    if isinstance(videos, (list, tuple)) and isinstance(
            videos[0], (list, tuple)) and is_valid_image(videos[0][0]):
        return videos

    elif isinstance(videos, (list, tuple)) and is_valid_image(videos[0]):
        if isinstance(videos[0], Image.Image) or len(videos[0].shape) == 3:
            return [videos]
        elif len(videos[0].shape) == 4:
            return [list(video) for video in videos]

    elif is_valid_image(videos) and len(videos.shape) == 4:
        return [list(videos)]

    raise ValueError(f"Could not make batched video from {videos}")


class LlavaOnevisionVideoProcessor1(BaseImageProcessor):

    model_input_names = ["pixel_values_videos"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[Dict[str, int]] = None,
        resample: F.InterpolationMode = F.InterpolationMode.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if size is None:
            size = {"height": 384, "width": 384}
        # size = size if size is not None else {"height": 384, "width": 384}
        size = get_size_dict(size, default_to_square=False)

        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.do_convert_rgb = do_convert_rgb

    def _preprocess(
        self,
        images: ImageInput,
        do_resize: bool = True,
        size: Optional[List[int]] = None,
        resample: PILImageResampling = None,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        if size is None:
            size = [384, 384]
        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]
        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]
        if is_scaled_image(images[0]) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled videos. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False`"
                " to avoid rescaling them again.")

        image_type = get_image_type(images[0])

        if image_type == ImageType.PIL:
            images = [F.pil_to_tensor(image) for image in images]
        if image_type == ImageType.NUMPY:
            images = [torch.from_numpy(image).contiguous() for image in images]

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])
        if input_data_format == ChannelDimension.LAST:
            images = [image.permute(2, 0, 1).contiguous() for image in images]
            input_data_format = ChannelDimension.FIRST

        if do_rescale and do_normalize:
            # fused rescale and normalize
            new_mean = torch.tensor(
                image_mean, device=images[0].device) * (1.0 / rescale_factor)
            new_std = torch.tensor(
                image_std, device=images[0].device) * (1.0 / rescale_factor)

        if do_resize:
            images = [
                F.resize(image, size=size, interpolation=resample)
                for image in images
            ]

        if do_rescale and do_normalize:
            images = [
                F.normalize(image.to(dtype=torch.float32),
                            mean=new_mean,
                            std=new_std) for image in images
            ]
        elif do_rescale:
            images = [image * rescale_factor for image in images]
        elif do_normalize:
            images = [
                F.normalize(image.to(dtype=torch.float32),
                            mean=image_mean,
                            std=image_std) for image in images
            ]

        # images = [
        #     to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
        #       for image in images
        # ]

        return torch.stack(images)

    def preprocess(
        self,
        videos: VideoInput,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample: F.InterpolationMode = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):

        do_resize = do_resize if do_resize is not None else self.do_resize
        if size is None:
            size = self.size
        # size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        videos = make_batched_videos(videos)

        if not valid_images(videos[0]):
            raise ValueError(
                "Invalid video type. Must be a list consisting of PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray.")

        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )
        if size is not None:
            size_tuple = [size["height"], size["width"]
                          ] if "height" in size and "width" in size else [
                              size["shortest_edge"], size["shortest_edge"]
                          ]

        pixel_values = [
            self._preprocess(
                video,
                do_resize=do_resize,
                size=size_tuple,
                resample=resample,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
                do_convert_rgb=do_convert_rgb,
                data_format=data_format,
                input_data_format=input_data_format,
            ) for video in videos
        ]
        pixel_values = torch.stack(pixel_values)

        feature = BatchFeature(
            data={"pixel_values_videos": pixel_values},
            tensor_type=None,
        )
        return feature


def init_fast_processor():
    AutoImageProcessor.register("LlavaOnevisionVideoProcessor1",
                                LlavaOnevisionVideoProcessor1)
