from unittest import TestCase

from onnx_transformers.model_for_encoding import ModelForEncoding


class ModelForEncodingTest(TestCase):

    def test_abc(self):
        encoder = ModelForEncoding("./dummy")
        encoder("This is an example text")
