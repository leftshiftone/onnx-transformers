import logging
from dataclasses import dataclass
from typing import Optional

from onnx_transformers.__spi__.model import TokenizerConfig

logger = logging.getLogger(__name__)


@dataclass
class SpecialTokensMixin:
    config: TokenizerConfig

    @property
    def cls_token_id(self) -> Optional[int]:
        """
        :obj:`Optional[int]`: Id of the classification token in the vocabulary, to extract a summary of an input
        sequence leveraging self-attention along the full depth of the model.

        Returns :obj:`None` if the token has not been set.
        """
        token = self.config.cls_token
        return token["id"] if token is not None else None

    @property
    def cls_token(self) -> str:
        """
        :obj:`str`: Classification token, to extract a summary of an input sequence leveraging self-attention along the
        full depth of the model. Log an error if used while not having been set.
        """
        token = self.config.cls_token
        return token["content"] if token is not None else None

    @property
    def bos_token(self) -> str:
        """
        :obj:`str`: Beginning of sentence token. Log an error if used while not having been set.
        """
        token = self.config.bos_token
        return token["content"] if token is not None else None

    @property
    def bos_token_id(self) -> Optional[int]:
        """
        :obj:`Optional[int]`: Id of the token which shows the beginning of sentence token. Log an error if used
        while not having been set.
        """
        token = self.config.bos_token
        return token["id"] if token is not None else None

    @property
    def eos_token(self) -> str:
        """
        :obj:`str`: End of sentence token. Log an error if used while not having been set.
        """
        token = self.config.eos_token
        return token["content"] if token is not None else None

    @property
    def eos_token_id(self) -> Optional[int]:
        """
        :obj:`Optional[int]`: The id of the End of sentence token. Log an error if used while not having been set.
        """
        token = self.config.eos_token
        return token["id"] if token is not None else None

    @property
    def unk_token(self) -> str:
        """
        :obj:`str`: Unknown token. Log an error if used while not having been set.
        """
        token = self.config.unk_token
        return token["content"] if token is not None else None

    @property
    def unk_token_id(self) -> Optional[int]:
        """
        :obj:`Optional[int]`: The id of the Unknown token. Log an error if used while not having been set.
        """
        token = self.config.unk_token
        return token["id"] if token is not None else None

    @property
    def sep_token(self) -> str:
        """
        :obj:`str`: Separation token, to separate context and query in an input sequence. Log an error if used while
        not having been set.
        """
        token = self.config.sep_token
        return token["content"] if token is not None else None

    @property
    def sep_token_id(self) -> Optional[int]:
        """
        :obj:`Optional[int]`: The id of the Separation token, to separate context and query in an input sequence.
        Log an error if used while not having been set.
        """
        token = self.config.sep_token
        return token["id"] if token is not None else None

    @property
    def pad_token(self) -> str:
        """
        :obj:`str`: Padding token. Log an error if used while not having been set.
        """
        token = self.config.pad_token
        return token["content"] if token is not None else None

    @property
    def pad_token_id(self) -> Optional[int]:
        """
        :obj:`Optional[int]`: The id of the Padding token. Log an error if used while not having been set.
        """
        token = self.config.pad_token
        return token["id"] if token is not None else None

    @property
    def pad_token_type_id(self) -> Optional[int]:
        """
        :obj:`Optional[int]`: The id of the type id token. Log an error if used while not having been set.
        """
        return 0
