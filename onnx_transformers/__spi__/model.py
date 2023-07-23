from abc import ABC
from collections import OrderedDict
from dataclasses import dataclass
from os.path import exists
from typing import Dict, List
from typing import Optional

from onnx_transformers.__spi__.enums import PaddingStrategy, TruncationStrategy
from onnx_transformers.__spi__.types import Direction
from onnx_transformers.__util__.file_util import from_json


#  Copyright 2022 The X-and-Y team
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


class TokenizerConfig(OrderedDict):
    special_tokens: Dict[str, int] = {}

    def __init__(self, data: {}):
        super().__init__(data)

    def _get_token(self, name: str, abbreviations: List[str]):
        if name in self:
            return self[name]

        for entry in self["added_tokens"]:
            for abbreviation in abbreviations:
                if entry["content"] == abbreviation:
                    return entry

        return None

    @property
    def cls_token(self):
        return self._get_token("cls_token", ["[CLS]"])

    @property
    def pad_token(self):
        return self._get_token("pad_token", ["[PAD]", "<pad>"])

    @property
    def sep_token(self):
        return self._get_token("sep_token", ["[SEP]"])

    @property
    def eos_token(self):
        return self._get_token("eos_token", ["[EOS]", "</s>"])

    @property
    def unk_token(self):
        return self._get_token("unk_token", ["[UNK]", "<unk>"])

    @property
    def bos_token(self):
        return self._get_token("bos_token", ["[BOS]"])

    @property
    def mask_token(self):
        return self._get_token("mask_token", ["[MASK]", "<mask>"])


@dataclass
class EncodingFast:
    """This is dummy class because without the `tokenizers` library we don't have these objects anyway"""
    pass


@dataclass
class ModelIO(ABC):
    """
    This data class contains all information about model input nodes.
    This information can be used by a tokenizer to prepare the model input.
    """
    names: List[str]
    types: Dict[str, str]
    shapes: Dict[str, List[str]]


@dataclass
class ModelInfo:
    group: str
    name: str
    version: str
    license: str
    size: int
    labels: List[str]
    quantization: str
    optimization: str
    opset_version: int
    exporter_version: str
    base_model: dict
    libraries: dict
    python: str

    @staticmethod
    def from_file(path: str):
        return ModelInfo(**from_json(path))


@dataclass
class NetworkInfo:
    """
    A Network info data class contains all information about input/output nodes.
    This information can be used by a tokenizer to prepare the model input/output.
    """
    input: ModelIO
    output: ModelIO

    def get_input_names(self) -> List[str]:
        return self.input.names

    def get_input_type(self, name: str) -> str:
        return self.input.types[name] if name in self.input.types else None

    def get_input_shape(self, name: str) -> List[str]:
        return self.input.shapes[name] if name in self.input.shapes else None

    def get_output_names(self) -> List[str]:
        return self.output.names

    def get_output_type(self, name: str) -> str:
        return self.output.types[name] if name in self.output.types else None

    def get_output_shape(self, name: str) -> List[str]:
        return self.output.shapes[name] if name in self.output.shapes else None


@dataclass
class EncodingConfig:
    add_special_tokens: bool = True
    is_split_into_words: bool = False
    return_token_type_ids: Optional[bool] = None
    return_attention_mask: Optional[bool] = None
    return_overflowing_tokens: bool = False
    return_special_tokens_mask: bool = False
    return_offsets_mapping: bool = False
    return_length: bool = False
    padding_strategy: PaddingStrategy = PaddingStrategy.LONGEST
    truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE
    max_length: Optional[int] = None
    stride: int = 0
    pad_to_multiple_of: Optional[int] = None
    padding_direction: Direction = Direction.RIGHT
    truncation_direction: Direction = Direction.RIGHT
    do_lower_case: Optional[bool] = None
    output_name: str = "embedding"

    @staticmethod
    def from_config(config_path: str):
        encoding_config = EncodingConfig()

        import json
        from os.path import isfile
        max_length_variations = [
            "max_len",
            "model_max_length",
            "model_max_len",
            "max_length"
        ]
        if isfile(config_path + "/tokenizer_init.json"):
            with open(config_path + "/tokenizer_init.json", "r", encoding="utf-8") as f:
                config = json.load(f)
                for max_len_var in max_length_variations:
                    if max_len_var in config:
                        encoding_config.max_length = config[max_len_var]
                        encoding_config.truncation_strategy = TruncationStrategy.LONGEST_FIRST
                        break
                if "do_lower_case" in config:
                    encoding_config.do_lower_case = config["do_lower_case"]

        return encoding_config


@dataclass
class ModelConfig(dict):

    @property
    def config_path(self):
        return self["config_path"]

    @staticmethod
    def from_path(config_path: str):
        assert exists(config_path + "/config.json"), f"no config file found at {config_path}"
        config = from_json(config_path + "/config.json")
        config["config_path"] = config_path
        model_config = ModelConfig()
        model_config.update(config)
        return model_config
