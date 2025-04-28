import argparse
import logging
import zipfile
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import sys
import subprocess
import openslide
import psutil
from pyhist import PySlide, TileGenerator
from src import utility_functions

# Log to stdout so Galaxy captures it
logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

def log_memory_usage():
    """Log current memory usage of the process."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logging.info(f"Memory usage: RSS={mem_info.rss/1024/1024:.2f} MB, VMS={mem_info.vms/1024/1024:.2f} MB")

def run_pyhist_direct(image_path: Path, output_dir: Path, original_name: str):
    """Process one image with PyHIST, then return the folder where tiles were written."""
    original_base = Path(original_name).stem

    # --- Pass the base output directory, not the tile folder itself ---
    output_root = output_dir / "output"

    logging.info(f"Processing image: {image_path}")
    log_memory_usage()

    # Validate input with OpenSlide
    try:
        slide_ = openslide.OpenSlide(str(image_path))
        logging.info(f"Input file validated with OpenSlide: {image_path}")
        slide_.close()
    except openslide.OpenSlideError as e:
        raise RuntimeError(f"Invalid input file: {e}") from e

    # Check segmentation binary
    segment_path = "/pyhist/src/graph_segmentation/segment"
    if os.path.exists(segment_path) and os.access(segment_path, os.X_OK):
        logging.info(f"Segmentation executable found: {segment_path}")
    else:
        logging.warning(f"Segmentation executable missing, will use Otsu method")

    # Build the PyHIST argument dict
    args_dict = {
        'svs': str(image_path),
        'patch_size': 256,
        'method': 'otsu',
        'thres': 0.1,
        'output_downsample': 8,
        'mask_downsample': 8,
        'borders': '0000',
        'corners': '1010',
        'pct_bc': 1,
        'k_const': 1000,
        'minimum_segmentsize': 1000,
        'save_patches': True,
        'save_blank': False,
        'save_nonsquare': False,
        'save_tilecrossed_image': False,
        'save_mask': True,
        'save_edges': False,
        'info': 'verbose',
        'output': str(output_root),
        'format': 'png',
    }

    # Run PyHIST
    utility_functions.check_image(args_dict['svs'])
    loglevel = {"default": logging.INFO, "verbose": logging.DEBUG, "silent": logging.CRITICAL}
    logging.getLogger().setLevel(loglevel[args_dict['info']])

    slide = PySlide(args_dict)
    logging.info(f"Slide loaded: {slide}")
    tile_extractor = TileGenerator(slide)
    logging.info(f"TileExtractor initialized: {tile_extractor}")

    try:
        tile_extractor.execute()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Tile extraction subprocess failed: {e}") from e

    # --- Grab the actual folder PyHIST used for tiles ---
    tile_dir = Path(slide.tile_folder)
    tiles = list(tile_dir.glob("*.png"))
    logging.info(f"Found {len(tiles)} tiles in {tile_dir}")

    utility_functions.clean(slide)
    return tile_dir

def append_to_zip(output_zip_path, original_name, tile_dir):
    """Append all .png tiles from tile_dir into the ZIP."""
    original_base = Path(original_name).stem
    with zipfile.ZipFile(output_zip_path, 'a', zipfile.ZIP_DEFLATED) as zipf:
        for file in tile_dir.glob("*.png"):
            num = file.stem.split("_")[-1]
            arcname = f"{original_base}/{original_base}_{num}.png"
            zipf.write(file, arcname)
    logging.info(f"Appended {len(list(tile_dir.glob('*.png')))} tiles to {output_zip_path}")

def process_image(args):
    """Wrapper to run one image & zip its tiles."""
    image_path_str, original_name, output_zip = args
    input_path = Path(image_path_str)
    output_zip_path = Path(output_zip)

    if not input_path.exists():
        logging.warning(f"Input file missing: {input_path}")
        return

    try:
        tile_dir = run_pyhist_direct(input_path, output_zip_path.parent, original_name)
        if tile_dir.exists() and any(tile_dir.glob("*.png")):
            append_to_zip(output_zip_path, original_name, tile_dir)
        else:
            logging.warning(f"No tiles to zip in {tile_dir}")
    except Exception as e:
        logging.error(f"Error processing {input_path}: {e}")
        raise

def main():
    os.chdir("/pyhist")
    logging.info(f"Working directory: {os.getcwd()}")

    parser = argparse.ArgumentParser(description="Tile extraction for Galaxy")
    parser.add_argument('--input', action='append', help="Input image paths", default=[])
    parser.add_argument('--original_name', action='append', help="Original file names", default=[])
    parser.add_argument('--output_zip', required=True, help="Output ZIP file path")
    args = parser.parse_args()

    if len(args.input) != len(args.original_name):
        raise ValueError("Mismatch between input paths and original names")

    max_workers = 1
    tasks = list(zip(args.input, args.original_name, [args.output_zip]*len(args.input)))

    # Create fresh ZIP and process images
    with zipfile.ZipFile(args.output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        logging.info(f"Created ZIP: {args.output_zip}")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for future in executor.map(process_image, tasks):
                pass

    # Clean up
    shutil.rmtree(Path(args.output_zip).parent / "output", ignore_errors=True)
    logging.info("Done. Temporary files cleaned up.")
    logging.info(f"Final ZIP size: {Path(args.output_zip).stat().st_size} bytes")

if __name__ == "__main__":
    main()
