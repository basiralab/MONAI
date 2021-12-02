# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np
import torch

from monai.config import DtypeLike, NdarrayOrTensor, PathLike
from monai.data.utils import compute_shape_offset, create_file_basename, to_affine_nd
from monai.networks.layers import AffineTransform
from monai.utils import GridSampleMode, GridSamplePadMode, optional_import, require_pkg
from monai.utils.type_conversion import convert_data_type

if TYPE_CHECKING:
    import itk  # type: ignore
    import nibabel as nib
    from nibabel.nifti1 import Nifti1Image
    from PIL import Image as PILImage

    has_itk = has_nib = has_pil = True
else:
    itk, has_itk = optional_import("itk", allow_namespace_pkg=True)
    nib, has_nib = optional_import("nibabel")
    Nifti1Image, _ = optional_import("nibabel.nifti1", name="Nifti1Image")
    PILImage, has_pil = optional_import("PIL.Image")

__all__ = ["ImageWriter", "ITKWriter", "NibabelWriter", "PILWriter", "FolderLayout"]


class FolderLayout:
    """
    A utility class to create organized filenames within ``output_dir``.

    Example:

    .. code-block:: python

        from monai.data import FolderLayout

        layout = FolderLayout(output_dir="/test_run_1/", postfix="seg", ext=".nii", makedirs=False)
        layout.filename(subject="Sub-A", idx="00", modality="T1")
        # return value: "/test_run_1/Sub-A_seg_00_modality-T1.nii"

    """

    def __init__(
        self,
        output_dir: PathLike,
        postfix: str,
        ext: str,
        parent: bool = False,
        makedirs: bool = False,
        overwrite: bool = False,
    ):
        """
        Args:
            output_dir: output directory.
            postfix: postfix for output file name (an underscore will be prepended).
            ext: output file extension to be appended to the end of an output filename.
            parent: whether to add a level of parent folder to contain each image to the output filename.
            makedirs: whether to create the output parent directories if they do not exist.
            overwrite: whether to overwrite existing files.
        """
        self.output_dir = output_dir
        self.postfix = postfix
        self.ext = ext
        self.parent = parent
        self.makedirs = makedirs
        self.overwrite = overwrite

    def filename(self, subject=None, idx=None, **kwargs):
        """
        Create a filename based on the input ``subject`` and ``idx``.

        The output filename is formed as:

            ``output_dir/subject/subject[_postfix][_idx][_key-value][ext]``

        Args:
            subject: subject name, used as the primary id of the output filename.
                When a PathLike object is provided, the base filename will be used as the subject name,
                the extension will be ignored, in favor of ``ext`` from this class.
            idx: additional index name of the image (an underscore will be prepended).
            kwargs: additional keyword arguments to be used to form the output filename.
        """
        full_name = create_file_basename(
            input_file_name=subject,
            postfix=self.postfix,
            folder_path=self.output_dir,
            separate_folder=self.parent,
            patch_index=idx,
            makedirs=self.makedirs,
        )
        for k, v in kwargs.items():
            full_name += f"_{k}-{v}"
        if self.ext is not None:
            full_name += f"{self.ext}"
        return full_name


class ImageWriter:
    @classmethod
    def data_obj(cls, data_array, metadata, **kwargs):
        raise NotImplementedError(f"Subclass of {cls.__name__} must implement this method.")

    @classmethod
    def write(cls, filename_or_obj, data_obj, **kwargs):
        raise NotImplementedError(f"Subclass of {cls.__name__} must implement this method.")


@require_pkg(pkg_name="itk")
class ITKWriter(ImageWriter):
    """
    Write data and metadata into files to disk using ITK-python.
    """

    @classmethod
    def write(cls, filename_or_obj, data_obj, **kwargs):
        itk.imwrite(data_obj, filename_or_obj, **kwargs)


@require_pkg(pkg_name="nibabel")
class NibabelWriter(ImageWriter):
    """
    Write data and metadata into files to disk using Nibabel.
    """

    @classmethod
    def data_obj(
        cls,
        data_array: NdarrayOrTensor,
        metadata: Optional[dict] = None,
        resample: bool = True,
        channel_dim: Optional[int] = 0,
        squeeze_end_dims: bool = True,
        **kwargs,
    ):
        """
        Create a Nibabel object based on ``data_array`` and ``metadata``.
        Different from ``NibabelWriter.create_data_obj``, this method allows for flexible
        data and metadata specifications.

        Spatially it supports up to three dimensions, that is, H, HW, HWD for
        1D, 2D, 3D respectively (with resampling supports for 2D and 3D only).

        When saving multiple time steps or multiple channels `data_array`,
        time and/or modality axes should be the at the `channel_dim`.
        For example, the shape of a 2D eight-class and channel_dim=0, the
        segmentation probabilities to be saved could be `(8, 64, 64)`;
        in this case ``data_array`` will be converted to a `(64, 64, 1, 8)`
        NIfTI object (the third dimension is reserved as a spatial dimension).

        The `metadata` could optionally have the following keys:

            - ``'original_affine'``: for data original affine, it will be the
              affine of the output Nifti1Image object, defaulting to an identity matrix.
            - ``'affine'``: it should specify the current data affine, defaulting to an identity matrix.
            - ``'spatial_shape'``: for data output spatial shape.

        When ``metadata`` is specified and ``resample=True``, the saver will try to
        resample batch data from the space defined by `"affine"` to the space
        defined by `"original_affine"`.

        Args:
            data_array: input data for ``Nifti1Image.data``.
            metadata: an optional dictionary of meta information.
            resample: whether to convert the data array to its original coordinate system
                based on `"original_affine"` in ``metadata``.
            channel_dim: specifies the axis of the data array that is the channel dimension.
                ``None`` means no channel dimension.
            squeeze_end_dims: if ``True``, any trailing singleton dimensions will be removed (after the channel
                has been moved to the end). So if input is (H,W,D,C) and C==1, then it will be saved as (H,W,D).
                If D is also 1, it will be saved as (H,W). If ``False``, image will always be saved as (H,W,D,C).
            kwargs: additional keyword arguments to pass to :py:func:`NibabelWriter.create_data_obj`.

        Returns: a ``Nifti1Image`` object.
        """
        original_affine, affine, spatial_shape = None, None, None
        if metadata:
            affine = metadata.get("affine")
            spatial_shape = metadata.get("spatial_shape")
            if resample:
                original_affine = metadata.get("original_affine")
        data_array = NibabelWriter.convert_data_array(data_array, channel_dim, squeeze_end_dims)
        return NibabelWriter.create_data_obj(
            data=data_array, affine=affine, target_affine=original_affine, output_spatial_shape=spatial_shape, **kwargs
        )

    @staticmethod
    def convert_data_array(data, channel_dim: Optional[int] = 0, squeeze_end_dims: bool = True):
        """
        Rearrange the data array axes to be compatible with the NIfTI format.

        Args:
            data: input data to be converted to "channel-last" format.
            channel_dim: specifies the axis of the data array that is the channel dimension.
                ``None`` means no channel dimension.
            squeeze_end_dims: if ``True``, any trailing singleton dimensions will be removed (after the channel
                has been moved to the end). So if input is (H,W,D,C) and C==1, then it will be saved as (H,W,D).
                If D is also 1, it will be saved as (H,W). If ``False``, image will always be saved as (H,W,D,C).
        """
        data_array, *_ = convert_data_type(data, dtype=np.ndarray)
        # change data to "channel last" format
        if channel_dim is not None:
            data_array = np.moveaxis(data_array, channel_dim, -1)
        else:  # adds a channel dimension
            data_array = data_array[..., np.newaxis]
        # change data shape to be (h, w, d, channel)
        while len(data_array.shape) < 4:
            data_array = np.expand_dims(data_array, -2)
        # if desired, remove trailing singleton dimensions
        while squeeze_end_dims and data_array.shape[-1] == 1:
            data_array = np.squeeze(data_array, -1)
        return data_array

    @staticmethod
    def create_data_obj(
        data: NdarrayOrTensor,
        affine: Optional[NdarrayOrTensor] = None,
        target_affine: Optional[np.ndarray] = None,
        output_spatial_shape: Union[Sequence[int], np.ndarray, None] = None,
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.BORDER,
        align_corners: bool = False,
        dtype: DtypeLike = np.float64,
        output_dtype: DtypeLike = np.float32,
    ):
        """
        Create a ``Nifti1Image`` object based on ``data`` array and
        ``target_affine``. This function converts data into the coordinate
        system defined by ``target_affine`` when ``target_affine`` is
        specified.

        If the coordinate transform between ``affine`` and ``target_affine``
        could be achieved by simply transposing and flipping `data`, no
        resampling will happen.  Otherwise this function resamples `data` using
        the transformation computed from ``affine`` and ``target_affine``. Note
        that the shape of the resampled ``data`` may subject to some rounding
        errors. For example, resampling a 20x20 pixel image from pixel size
        (1.5, 1.5)-mm to (3.0, 3.0)-mm space will return a 10x10-pixel image.
        However, resampling a 20x20-pixel image from pixel size (2.0, 2.0)-mm
        to (3.0, 3.0)-mm space will output a 14x14-pixel image, where the image
        shape is rounded from 13.333x13.333 pixels. In this case
        `output_spatial_shape` could be specified so that this function writes
        image data to a designated shape.

        The saved ``affine`` matrix follows:

            - If ``affine`` equals to ``target_affine``, save the data with ``target_affine``.
            - If ``resample=False``, transform ``affine`` to a new affine based on
              the orientation of ``target_affine`` and save the data with the new affine.
            - If ``resample=True``, save the data with ``target_affine``, if
              explicitly specify the ``output_spatial_shape``, the shape of saved
              data is not computed by ``target_affine``.
            - If ``target_affine`` is None, set ``target_affine=affine`` and save.
            - If ``affine`` and ``target_affine`` are ``None``, the data will be saved
              with an identity matrix as the image affine.

        Different from ``self.data_obj``, this function assumes the NIfTI
        dimension notations. Spatially it supports up to three dimensions, that
        is, H, HW, HWD for 1D, 2D, 3D respectively. When saving multiple time
        steps or multiple channels `data`, time and/or modality axes should be
        appended after the first three dimensions. For example, shape of 2D
        eight-class segmentation probabilities to be saved could be `(64, 64,
        1, 8)`. Also, data in shape (64, 64, 8), (64, 64, 8, 1) will be
        considered as a single-channel 3D image.

        Args:
            data: input data to write to file.
            affine: the current affine of `data`. Defaults to `np.eye(4)`
            target_affine: before saving the (`data`, `affine`) as `Nifti1Image`,
                transform the data into the coordinates defined by `target_affine`.
            output_spatial_shape: spatial shape of the output image.
                This option is used when resample = True.
            mode: {``"bilinear"``, ``"nearest"``}
                This option is used when ``resample = True``.
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                This option is used when ``resample = True``.
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            align_corners: Geometrically, we consider the pixels of the input
                as squares rather than points.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            dtype: data type for resampling computation. Defaults to
                ``np.float64`` for best precision. If None, use the data type of input data.
            output_dtype: data type for saving data. Defaults to ``np.float32``.
        """
        data, *_ = convert_data_type(data, np.ndarray)
        affine = to_affine_nd(3, affine) if affine is not None else np.eye(4, dtype=np.float64)
        target_affine = to_affine_nd(3, target_affine) if target_affine is not None else affine
        if np.allclose(affine, target_affine, atol=1e-3):
            # no affine changes, return (data, affine)
            return nib.Nifti1Image(data.astype(output_dtype, copy=False), to_affine_nd(3, target_affine))

        # resolve orientation
        start_ornt = nib.orientations.io_orientation(affine)
        target_ornt = nib.orientations.io_orientation(target_affine)
        ornt_transform = nib.orientations.ornt_transform(start_ornt, target_ornt)
        data = nib.orientations.apply_orientation(data, ornt_transform)
        _affine = affine @ nib.orientations.inv_ornt_aff(ornt_transform, data.shape)
        if np.allclose(_affine, target_affine, atol=1e-3):
            return nib.Nifti1Image(data.astype(output_dtype, copy=False), to_affine_nd(3, _affine))

        # need resampling
        dtype = dtype or data.dtype
        if output_spatial_shape is None:
            output_spatial_shape, _ = compute_shape_offset(data.shape, _affine, target_affine)
        output_spatial_shape_ = list(output_spatial_shape) if output_spatial_shape is not None else []
        sp_dims = min(data.ndim, 3)
        output_spatial_shape_ += [1] * (sp_dims - len(output_spatial_shape_))
        output_spatial_shape_ = output_spatial_shape_[:sp_dims]
        original_channels = data.shape[3:]
        if original_channels:  # multi channel, resampling each channel
            data_np: np.ndarray = data.reshape(list(data.shape[:3]) + [-1])  # type: ignore
            data_np = np.moveaxis(data_np, -1, 0)  # channel first for pytorch
        else:  # single channel image, need to expand to have a channel
            data_np = data[None]
        affine_xform = AffineTransform(
            normalized=False, mode=mode, padding_mode=padding_mode, align_corners=align_corners, reverse_indexing=True
        )
        data_torch = affine_xform(
            torch.as_tensor(np.ascontiguousarray(data_np, dtype=dtype)).unsqueeze(0),
            torch.as_tensor(np.ascontiguousarray(np.linalg.inv(_affine) @ target_affine, dtype=dtype)),
            spatial_size=output_spatial_shape_,
        )
        data_np = data_torch[0].detach().cpu().numpy()
        if original_channels:
            data_np = np.moveaxis(data_np, 0, -1)  # channel last for nifti
            data_np = data_np.reshape(list(data_np.shape[:3]) + list(original_channels))
        else:
            data_np = data_np[0]
        return nib.Nifti1Image(data_np.astype(output_dtype, copy=False), to_affine_nd(3, target_affine))

    @classmethod
    def write(cls, filename_or_obj, data_obj, **kwargs):
        nib.save(data_obj, filename_or_obj)


@require_pkg(pkg_name="PIL")
class PILWriter(ImageWriter):
    pass
