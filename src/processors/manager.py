import inspect
import pkgutil
from src.logger import logger

class Processor:
    # Base class that each processor must inherit from.
    def __init__(self):
        self.description = "UNKNOWN"


class ProcessorManager:
    
    def __init__(self, processors_dir="src.processors"):
        self.processors_dir = processors_dir
        self.reload_processors()

    def reload_processors(self):
        self.processors = {}
        self.seen_paths = []

        logger.info(f'Looking for processors in "{self.processors_dir}"')
        self.walk_package(self.processors_dir)

    @staticmethod
    def get_name_filter(processor_name):
        def filter_function(member):
            return inspect.isclass(member) and member.__module__ == processor_name

        return filter_function

    def walk_package(self, package):
        # walk the supplied package to retrieve all processors
        imported_package = __import__(package, fromlist=["blah"])
        loaded_packages = []
        for _, processor_name, ispkg in pkgutil.walk_packages(
            imported_package.__path__, imported_package.__name__ + "."
        ):
            if not ispkg and processor_name != __name__:
                processor_module = __import__(processor_name, fromlist=["blah"])
                clsmembers = inspect.getmembers(
                    processor_module,
                    ProcessorManager.get_name_filter(processor_name),
                )
                for (_, c) in clsmembers:
                    # Only add classes that are a sub class of Processor, but NOT Processor itself
                    if issubclass(c, Processor) & (c is not Processor):
                        self.processors[c.__name__] = c
                        loaded_packages.append(c.__name__)

        logger.info(f"Loaded processors: {loaded_packages}")
