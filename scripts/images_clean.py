import os
import cv2
import progressbar
import shutil
import argparse
import concurrent.futures
import glob
import jpeg4py as jpeg

parser = argparse.ArgumentParser(description='Cleans invalid images')
parser.add_argument('--data-path', required=True, help='Path where the images are stored')
parser.add_argument('--formats', nargs='+', default=['jpg', 'jpeg', 'png'], help='Image formats to search in the path')
parser.add_argument('--remove-sequence', action='store_false', help='Whether or not to remove entire sequence')
parser.add_argument('--max-workers', type=int, default=10, help='Number of workers to use')
parser.add_argument('--min-index', type=int, default=-1, help='Min index to continue the script from')
args = parser.parse_args()


def verify_sequence(sequence_path, bar, i, remove_sequence=False):
    # Create a list of paths of the images inside sequence_path
    images_paths = []
    for ext in args.formats:
        images_paths += glob.glob(os.path.join(sequence_path, '*.{}'.format(ext)))

    # Initialize the size of the images contained in sequence_path
    image_size = None

    # Iterate over every image to check that there are no loading errors
    for image_path in images_paths:
        try:
            # Get the file extension to decide which tool to use
            image_extension = os.path.splitext(image_path)[-1].replace('.', '')
            if image_extension in ['jpg', 'jpeg']:
                image = jpeg.JPEG(image_path).decode()
            elif image_extension in ['png']:
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            else:
                raise ValueError('Format not known.')

            # Check image is not None
            if image is None:
                raise ValueError('Image is None')

            # Check that the proportion of the image is correct
            if image.shape[0] > image.shape[1]:
                raise ValueError('Image size is not rectangular')

            # Verify that the image size is the same in all images inside a folder
            if image_size is not None and image.shape != image_size:
                raise ValueError('Image size is not the same')
            elif image_size is None:
                image_size = image.shape

        except Exception as e:
            print('Incorrect image {} - {}'.format(image_path, str(e)))
            if remove_sequence:
                shutil.rmtree(sequence_path)
                break
            else:
                os.remove(image_path)

    # Update bar counter
    if bar.value < i:
        bar.update(i)


# Create ThreadPool for parallel execution
executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers)

# Generate a list of sequences
folder_paths = sorted([root for root, _, _ in os.walk(args.data_path)])

# Create progress bar
bar = progressbar.ProgressBar(max_value=len(folder_paths))

# Walk through the folders of args.data_path
for i, folder_path in enumerate(folder_paths):
    if args.min_index != -1 and i < args.min_index:
        continue
    #executor.submit(verify_sequence, folder_path, bar, i, args.remove_sequence)
    verify_sequence(folder_path, bar, i, True)
