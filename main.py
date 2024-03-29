from logger import logger
from src.core import entry_point

logger.info(f"Loading OMRChecker modules...")

import argparse
from pathlib import Path

# construct the argument parse and parse the arguments
argparser = argparse.ArgumentParser()

argparser.add_argument(
    "-i",
    "--inputDir",
    default=["inputs"],
    nargs="*",
    required=False,
    type=str,
    dest="input_paths",
    help="Specify an input directory.",
)

argparser.add_argument(
    "-o",
    "--outputDir",
    default="outputs",
    required=False,
    dest="output_dir",
    help="Specify an output directory.",
)

argparser.add_argument(
    "-a",
    "--autoAlign",
    required=False,
    dest="autoAlign",
    action="store_true",
    help="(experimental) Enables automatic template alignment - \
    use if the scans show slight misalignments.",
)

argparser.add_argument(
    "-l",
    "--setLayout",
    required=False,
    dest="setLayout",
    action="store_true",
    help="Set up OMR template layout - modify your json file and \
    run again until the template is set.",
)


(
    args,
    unknown,
) = argparser.parse_known_args()
args = vars(args)

if len(unknown) > 0:
    logger.warning("".join(["\nError: Unknown arguments: ", unknown]))
    argparser.print_help()
    exit(11)

for root in args["input_paths"]:
    entry_point(
        Path(root),
        Path(root),
        args,
    )
