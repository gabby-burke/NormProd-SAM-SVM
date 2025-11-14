import sys
import ast
from process_image_pairs import process_image_pair, compute_normprod, nan_safe_mean_filter, compute_local_std, compute_boxcar_diff, compute_dog, fill_nans, check_raster_stats, cleanup_intermediate_files

base1 = sys.argv[1]
base2 = sys.argv[2]
data_dir = sys.argv[3]
epsg = int(sys.argv[4])
windows = ast.literal_eval(sys.argv[5])

process_image_pair(base1, base2, data_dir, epsg, windows)
