from typing import Optional

import numpy as np

from onnx_transformers.__spi__.model import EncodingConfig
from onnx_transformers.__spi__.types import Batch
from onnx_transformers.__util__.array_util import sigmoid
from onnx_transformers.model_for_encoding import ModelForEncoding


class ModelForCrossEncoding(ModelForEncoding):
    """
    This class is used as a wrapper of cross-encoding ONNX models.
    The output of the encode method is either the logits value or a probability calculated by sigmoid.
    """

    def __init__(self, config_path: str, use_sigmoid: bool = True):
        super().__init__(config_path)
        self.use_sigmoid = use_sigmoid

    def __call__(self, batch: Batch, config: Optional[EncodingConfig] = None):
        logits = np.squeeze(self.__call__(batch, config))
        return sigmoid(logits) if self.use_sigmoid else logits
