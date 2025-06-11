from pathlib import Path
from unittest import TestCase

from onnx_transformers.model_for_encoding import ModelForEncoding


class ModelForEncodingTest(TestCase):
    def test_abc(self):
        path = Path(__file__).parent / "dummy"
        encoder = ModelForEncoding(str(path))
        encoder("This is an example text")
