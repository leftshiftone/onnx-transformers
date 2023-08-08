import logging
from typing import Optional, Union, List

import numpy as np

from onnx_transformers.__spi__.model import NetworkInfo, EncodingConfig, ModelConfig
from onnx_transformers.abstract_model import AbstractModel
from onnx_transformers.tokenizer.tokenizer import FastTokenizer


class ModelForEncoding(AbstractModel):
    """
    This class is used as a wrapper of text embedding ONNX models.
    """

    def __init__(self, config_path: str):
        super().__init__(config_path)

        info = NetworkInfo(self.get_model_input(), self.get_model_output())
        self.tokenizer = FastTokenizer(ModelConfig.from_path(config_path), info)
        self.enc_config = EncodingConfig.from_config(config_path)

    def __call__(self, batch: Union[str, List[str]], config: Optional[EncodingConfig] = None):
        config = config if config else self.enc_config
        squeeze = not isinstance(batch, list)

        inputs = self.tokenizer.encode(batch, config)
        outputs = self.execute(inputs)

        if len(outputs) == 1:
            return np.squeeze(outputs[0]) if squeeze else outputs[0]

        output_names = self.get_model_output().names
        outputs = {k: v for k, v in zip(output_names, outputs)}

        return np.squeeze(outputs[config.output_name]) if squeeze else outputs[config.output_name]

    def get_output_dim(self) -> int:
        """
        Returns the output dimension of the onnx model. Some models cannot infer the output shape due to unsupported
        shape inferences. In this case the method tries to resolve the output dimension from the model config.
        :return: int
        """

        output_name = list(self.get_model_output().shapes.keys())[0]
        shape = self.get_model_output().shapes[output_name][1]

        if isinstance(shape, str):
            logging.warning("cannot determine output shape from onnx session. try to return dim from config")
            assert "dim" in self.model_config, "cannot determine output dim"
            return self.model_config.get("dim")
        # noinspection PyTypeChecker
        return shape
