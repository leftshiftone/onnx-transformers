from abc import ABC

# noinspection PyUnresolvedReferences
from onnxruntime import SessionOptions, InferenceSession

from onnx_transformers.__spi__.model import ModelIO, ModelInfo, ModelConfig


class AbstractModel(ABC):
    """
    ONNX Engine implementation.
    """

    def __init__(self, config_path: str, model_path: str = "model"):
        info = ModelInfo.from_file(config_path + "/info.json")
        model_path = config_path + "/" + model_path + ".onnx"

        self.model_config = ModelConfig.from_path(config_path)

        opt = SessionOptions()
        # opt.enable_profiling = True

        provider = "CUDAExecutionProvider" if info.optimization == "cuda" else "CPUExecutionProvider"
        self.session = InferenceSession(model_path, opt, providers=[provider])
        self.use_io_binding = "CUDAExecutionProvider" in self.session.get_providers()

    def execute(self, inputs: dict):
        """
        Runs the ONNX model with the given inputs.
        :param inputs: the inputs to run the model with
        :return: dict
        """

        if self.use_io_binding:
            io_binding = self.session.io_binding()
            for k in inputs:
                io_binding.bind_cpu_input(k, inputs[k])
            for k in self.get_model_output().names:
                io_binding.bind_output(k)
            self.session.run_with_iobinding(io_binding)
            return io_binding.copy_outputs_to_cpu()

        return self.session.run(None, inputs)

    def get_model_input(self) -> ModelIO:
        """
        Returns the model input information.
        :return: ModelIO
        """

        names = [x.name for x in self.session.get_inputs()]
        types = {x.name: x.type for x in self.session.get_inputs()}
        shapes = {x.name: x.shape for x in self.session.get_inputs()}

        return ModelIO(names, types, shapes)

    def get_model_output(self) -> ModelIO:
        """
        Returns the model output information.
        :return: ModelIO
        """

        names = [x.name for x in self.session.get_outputs()]
        types = {x.name: x.type for x in self.session.get_outputs()}
        shapes = {x.name: x.shape for x in self.session.get_outputs()}

        return ModelIO(names, types, shapes)
