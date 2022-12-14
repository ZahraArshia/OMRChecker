from src.processors.manager import Processor

class ImagePreprocessor(Processor):

    def __init__(self, options, args):
        self.options = options
        self.args = args
        super().__init__()

    def apply_filter(self, image, filename):
        raise NotImplementedError

    @staticmethod
    def exclude_files():
        return []
