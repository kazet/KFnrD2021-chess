from loader import Preprocessor, StandardConvSuite
import torch


class INFPreprocessor(Preprocessor):
    def __init__(self, device: torch.device):
        """
        Wrapper for `Iterator` class that preprocesses compressed data to pytorch tensors.
        :param device: device for tensors
        """
        super(Preprocessor, self).__init__()
        self.device = device


class Inference (StandardConvSuite):
    def __init__(self, device: torch.device, model):
        """
        Preprocesses dictionary of tensors to conv-suitable planes
        :param device: device for tensors
        :param model: model to predict
        """
        self.preprocessor = INFPreprocessor(device)
        super().__init__(self.preprocessor)
        self.coder = model
        self.coder.eval()

    def predict(self, fens):
        """
        :param fens: list of chess positions in fen notation
        :return: tensor
        """
        batch = self.preprocess_batch(self.preprocessor.preprocess_batch(fens))
        return self.coder(batch)
