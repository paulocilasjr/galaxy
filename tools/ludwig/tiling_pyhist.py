import os
import zipfile
import tempfile
import subprocess
import shutil
from pathlib import Path
from multiprocessing import Pool
from math import ceil

# Configure logging
import logging
logging.basicConfig(
    filename="tile_processing.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.svs'}
BATCH_SIZE_PERCENTAGE = 0.2  # 20% of total images per batch

def extract_zip(zip_file):
    """Extracts a ZIP file into a temporary directory."""
    temp_dir = tempfile.mkdtemp(prefix="zip_extract_")
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            file_list = [os.path.join(temp_dir, f) for f in zip_ref.namelist()]
        logging.info(f"ZIP file extracted to: {temp_dir}")
        return temp_dir, file_list
    except zipfile.BadZipFile as exc:
        raise RuntimeError("Invalid ZIP file.") from exc

def pull_docker_image():
    """Pulls the mmunozag/pyhist docker image if not already present."""
    try:
        subprocess.run(["docker", "pull", "mmunozag/pyhist"], check=True, capture_output=True, text=True)
        logging.info("Pulled docker image: mmunozag/pyhist")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to pull docker image: {e.stderr}")
        raise RuntimeError(f"Failed to pull mmunozag/pyhist: {e.stderr}") from e

def run_pyhist_docker(image_path, process_id=None):
    """Runs PyHIST via docker on the given image with specific parameters."""
    image_path = os.path.abspath(image_path)
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    ext = os.path.splitext(image_path)[1]
    current_dir = os.getcwd()

    container_image_path = f"/pyhist/images/{image_name}{ext}"
    output_subdir = f"temp_output_{process_id}" if process_id is not None else "output"

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{current_dir}:/pyhist/images",
        "mmunozag/pyhist",
        "--patch-size", "512",
        "--content-threshold", "0.4",
        "--output-downsample", "4",
        "--borders", "0000",
        "--corners", "1010",
        "--percentage-bc", "1",
        "--k-const", "1000",
        "--minimum_segmentsize", "1000",
        "--save-patches",
        "--save-tilecrossed-image",
        "--info", "verbose",
        "--output", f"{output_subdir}/",
        container_image_path
    ]

    logging.info(f"Running docker command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.info(f"PyHIST docker executed successfully for {image_path}")
        logging.debug(f"PyHIST stdout: {result.stdout}")
        logging.debug(f"PyHIST stderr: {result.stderr}")
        tile_dir = os.path.join(current_dir, output_subdir, image_name, f"{image_name}_tiles")
        return image_name, tile_dir
    except subprocess.CalledProcessError as e:
        logging.error(f"PyHIST docker failed for {image_path}: {e.stderr}")
        return image_name, None

def process_batch(batch, process_id):
    """Processes a batch of images in parallel."""
    image_tile_map = {}
    for image_path in batch:
        image_name, tile_dir = run_pyhist_docker(image_path, process_id)
        if tile_dir and os.path.exists(tile_dir) and any(f.endswith('.png') for f in os.listdir(tile_dir)):
            image_tile_map[image_name] = tile_dir
        else:
            logging.warning(f"Process {process_id}: No tiles generated for {image_name} at {tile_dir}")
    return image_tile_map

def process_files(input_path):
    """Processes either a ZIP file or a single image using PyHIST docker, with parallelization for ZIP."""
    temp_dir = tempfile.mkdtemp(prefix="process_")
    image_tile_map = {}
    
    try:
        ext = os.path.splitext(input_path)[1].lower()
        if ext in VALID_EXTENSIONS:
            # Single image: no parallelization
            image_name, tile_dir = run_pyhist_docker(input_path)
            if tile_dir and os.path.exists(tile_dir) and any(f.endswith('.png') for f in os.listdir(tile_dir)):
                image_tile_map[image_name] = tile_dir
            else:
                logging.warning(f"No tiles generated for {image_name} at {tile_dir}")
        elif ext == '.zip':
            # ZIP file: extract and process in parallel
            temp_dir, file_list = extract_zip(input_path)
            original_dir = os.getcwd()
            os.chdir(temp_dir)  # Change to temp_dir for ZIP processing
            try:
                valid_images = [f for f in file_list if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS]
                if not valid_images:
                    raise ValueError("No valid image files found in the ZIP")
                
                total_images = len(valid_images)
                batch_size = max(1, ceil(total_images * BATCH_SIZE_PERCENTAGE))  # At least 1 image per batch
                num_batches = ceil(total_images / batch_size)  # Ensure all images are covered
                batches = [valid_images[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
                logging.info(f"Total images: {total_images}, Batch size: {batch_size}, Number of batches: {len(batches)}")
                
                # Verify all images are covered
                covered_images = sum(len(batch) for batch in batches)
                if covered_images != total_images:
                    raise ValueError(f"Batching error: {covered_images} images covered, expected {total_images}")

                with Pool(processes=min(num_batches, os.cpu_count())) as pool:
                    results = pool.starmap(process_batch, [(batch, i) for i, batch in enumerate(batches)])
                
                for result in results:
                    image_tile_map.update(result)
            finally:
                os.chdir(original_dir)  # Restore original directory
        else:
            raise ValueError(f"Unsupported input file type: {ext}. Expected .zip or {VALID_EXTENSIONS}")
        
        return temp_dir, image_tile_map
    except Exception as e:
        raise e
    finally:
        for dir_name in os.listdir(os.getcwd()):
            if dir_name.startswith("temp_output_"):
                shutil.rmtree(os.path.join(os.getcwd(), dir_name), ignore_errors=True)
        if ext == '.zip':
            shutil.rmtree(temp_dir, ignore_errors=True)
        elif ext in VALID_EXTENSIONS:
            shutil.rmtree(os.path.join(os.getcwd(), "output"), ignore_errors=True)

def create_output_zip(image_tile_map, output_zip_path):
    """Creates a ZIP file containing tiles organized by image name."""
    with zipfile.ZipFile(output_zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
        for image_name, tile_dir in image_tile_map.items():
            for root, _, files in os.walk(tile_dir):
                for file in files:
                    if file.endswith('.png'):
                        file_path = os.path.join(root, file)
                        arcname = os.path.join(image_name, os.path.basename(file_path))
                        zipf.write(file_path, arcname)
                        logging.info(f"Added {file_path} to ZIP as {arcname}")
            logging.info(f"Added tiles for {image_name} to ZIP")
    logging.info(f"Output ZIP created: {output_zip_path}")

def main(input_path, output_zip_path):
    """Main function to process input (ZIP or single image) and create a tiled output ZIP."""
    pull_docker_image()
    temp_dir, image_tile_map = process_files(input_path)
    try:
        create_output_zip(image_tile_map, output_zip_path)
    finally:
        logging.info(f"Temporary directory cleaned up: {temp_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tile images from a ZIP file or single image using PyHIST docker.")
    parser.add_argument("--zip_file", required=True, help="Path to the input ZIP file or single image.")
    parser.add_argument("--output_zip", required=True, help="Path to the output ZIP file with tiles.")
    
    args = parser.parse_args()
    main(args.zip_file, args.output_zip)
