# initial setup
import os
import gc
import re 
import yaml
import shutil
from shapely import Polygon, box
import numpy as np
import pyproj
import rasterio
import rasterio.merge
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, Resampling

#import Tony_FastIce24
import json
import subprocess
from osgeo import gdal,ogr,osr,gdal_array, gdalconst
gdal.UseExceptions()
import scipy.ndimage
from scipy.ndimage import uniform_filter#, convolve   # ALEX ADDED THIS to get the local SD within the kernel
from scipy.ndimage import gaussian_filter, generic_filter  # And chat added this because it's faster than the two above
import pdb # Another Alex addition - for debugging.
import datetime
from pathlib import Path
import psutil
from datetime import datetime


def check_raster_stats(filepath):
    """Prints min, max, mean, and std of a raster file."""
    if not os.path.exists(filepath):
        print(f"❌ ERROR: {filepath} does not exist.")
        return
    
    ds = gdal.Open(filepath, gdal.GA_ReadOnly)
    if ds is None:
        print(f"❌ ERROR: Cannot open {filepath}")
        return
    
    arr = ds.GetRasterBand(1).ReadAsArray()
    print(f"📊 Stats for {filepath}:")
    print(f"  Min: {np.nanmin(arr)}")
    print(f"  Max: {np.nanmax(arr)}")
    print(f"  Mean: {np.nanmean(arr)}")
    print(f"  Std Dev: {np.nanstd(arr)}\n")
    ds = None

def fill_nans(image):
    """Fills NaNs before filtering to prevent them from growing."""
    nan_mask = np.isnan(image)
    
    if not np.any(nan_mask):  # If no NaNs, return original
        return image

    print("🔧 Filling NaNs before filtering...")

    # Simple approach: replace NaNs with the local mean of valid pixels
    mean_value = np.nanmean(image)
    image_filled = np.where(nan_mask, mean_value, image)

    return image_filled

# DoG is waaaaay wider than sigma. Like if sigma is 11 is goes out to like 33... Bad effects on the Tony algorithm. 
def compute_dog(image_path, output_path, sigma):
    """Computes the Difference of Gaussians (DoG) while preventing NaN spread."""
    if os.path.exists(output_path):
        print(f"✅ Skipping, {output_path} already exists.")
        return output_path

    ds = gdal.Open(image_path, gdal.GA_ReadOnly)
    if ds is None:
        print(f"❌ ERROR: Cannot open {image_path}")
        return None

    driver = gdal.GetDriverByName("GTIFF")

    # Read the single-band raster
    band = ds.GetRasterBand(1).ReadAsArray()

    # Fill NaNs before applying Gaussian filter
    band_filled = fill_nans(band)

    # Compute Gaussian smoothing on the filled data
    smoothed = gaussian_filter(band_filled, sigma=sigma)

    # DoG = Original - Smoothed
    dog = band - smoothed

    # Save DoG image
    out_ds = driver.Create(output_path, ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Float32,
                           options=["COMPRESS=DEFLATE", "BIGTIFF=YES"])
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())
    out_ds.GetRasterBand(1).WriteArray(dog)
    out_ds.GetRasterBand(1).SetNoDataValue(np.nan)
    out_ds.FlushCache()
    out_ds = None

    ds = None
    print(f"✅ Saved DoG image: {output_path}")
    return output_path

# Hmmmm it seems DoG is quite different from Tony's original boxcar.... Let's define a boxcar filter too.
def compute_boxcar_diff(image_path, output_path, width):
    """Computes the difference from 2D boxcar smoothing (boxcar DoG-like) while preventing NaN spread."""
    if os.path.exists(output_path):
        print(f"✅ Skipping, {output_path} already exists.")
        return output_path

    ds = gdal.Open(image_path, gdal.GA_ReadOnly)
    if ds is None:
        print(f"❌ ERROR: Cannot open {image_path}")
        return None

    driver = gdal.GetDriverByName("GTIFF")

    band = ds.GetRasterBand(1).ReadAsArray()

    # Fill NaNs before filtering
    band_filled = fill_nans(band)

    # Apply 2D boxcar filter
    smoothed = uniform_filter(band_filled, size=width, mode="nearest")

    # Difference from smoothed version
    diff = band - smoothed

    # Save result
    out_ds = driver.Create(output_path, ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Float32,
                           options=["COMPRESS=DEFLATE", "BIGTIFF=YES"])
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())
    out_ds.GetRasterBand(1).WriteArray(diff)
    out_ds.GetRasterBand(1).SetNoDataValue(np.nan)
    out_ds.FlushCache()
    out_ds = None

    ds = None
    print(f"✅ Saved boxcar diff image: {output_path}")
    return output_path

def compute_local_std(image_path, output_path, width):
    """Computes the local standard deviation in a boxcar window of given width."""
    if os.path.exists(output_path):
        print(f"✅ Skipping, {output_path} already exists.")
        return output_path

    ds = gdal.Open(image_path, gdal.GA_ReadOnly)
    if ds is None:
        print(f"❌ ERROR: Cannot open {image_path}")
        return None

    driver = gdal.GetDriverByName("GTIFF")

    band = ds.GetRasterBand(1).ReadAsArray()

    # Fill NaNs before filtering
    band_filled = fill_nans(band)

    # Compute mean and mean of squares in local window
    local_mean = uniform_filter(band_filled, size=width, mode="nearest")
    local_mean_sq = uniform_filter(band_filled**2, size=width, mode="nearest")

    # Standard deviation: sqrt(E[x^2] - (E[x])^2)
    local_std = np.sqrt(local_mean_sq - local_mean**2)

    # Save result
    out_ds = driver.Create(output_path, ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Float32,
                           options=["COMPRESS=DEFLATE", "BIGTIFF=YES"])
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())
    out_ds.GetRasterBand(1).WriteArray(local_std)
    out_ds.GetRasterBand(1).SetNoDataValue(np.nan)
    out_ds.FlushCache()
    out_ds = None

    ds = None
    print(f"✅ Saved local std image: {output_path}")
    return output_path

def nan_safe_mean_filter(values):
    """Computes the mean of non-NaN values."""
    valid_values = values[~np.isnan(values)]
    return np.nan if valid_values.size == 0 else np.mean(valid_values)


def compute_normprod(dog1, dog2, std1, std2, output_path, output_path_std, output_path_stdmean, output_path_smovar, window_size):
    """Computes Normalized Product (NormProd) using precomputed smoothed images, with NaN-safe summation."""
    print("In compute_normprod...")
    
    if os.path.exists(output_path):
        print(f"✅ Skipping, {output_path} already exists.")
        return output_path

    ds_dog1 = gdal.Open(dog1, gdal.GA_ReadOnly)
    ds_dog2 = gdal.Open(dog2, gdal.GA_ReadOnly)
    ds_std1 = gdal.Open(std1, gdal.GA_ReadOnly)
    ds_std2 = gdal.Open(std2, gdal.GA_ReadOnly)

    if not all([ds_dog1, ds_dog2, ds_std1, ds_std2]):
        print(f"❌ ERROR: Missing input files.")
        return None

    driver = gdal.GetDriverByName("GTIFF")

    print("Reading these in..")
    # Read the single-band rasters
    dog1 = ds_dog1.GetRasterBand(1).ReadAsArray()
    dog2 = ds_dog2.GetRasterBand(1).ReadAsArray()
    std1 = ds_std1.GetRasterBand(1).ReadAsArray()
    std2 = ds_std2.GetRasterBand(1).ReadAsArray()    

    # Mean the two stds together (to start with. GABBY you might have better ideas?)
    stdmean = np.mean(np.stack([std1, std2], axis=0), axis=0)

    
    
    print("Saving StdMean...")
    # Save output
    out_ds = driver.Create(output_path_stdmean, ds_dog1.RasterXSize, ds_dog1.RasterYSize, 1, gdal.GDT_Float32,
                           options=["COMPRESS=DEFLATE", "BIGTIFF=YES"])
    out_ds.SetGeoTransform(ds_dog1.GetGeoTransform())
    out_ds.SetProjection(ds_dog1.GetProjection())
    out_ds.GetRasterBand(1).WriteArray(stdmean)
    out_ds.GetRasterBand(1).SetNoDataValue(np.nan)
    out_ds.FlushCache()
    out_ds = None

    # make another variable with this squared: (call it variance)
    variance = stdmean*stdmean
    variance_filled = fill_nans(variance)
    # we can now remove stdmean from memory:
    stdmean = None
    # Here we should convolve (smooth) with a window_size*window_size boxcar
    # for reference, from above, here is how to do a boxcar smooth:
    #     smoothedVariance = uniform_filter(band_filled, size=width, mode="nearest")
    SmoothedVariance = uniform_filter(variance_filled, size = window_size, mode = "nearest")
    print("Saving Smoothed Variance...")
    # Save output
    out_ds = driver.Create(output_path_smovar, ds_dog1.RasterXSize, ds_dog1.RasterYSize, 1, gdal.GDT_Float32,
                           options=["COMPRESS=DEFLATE", "BIGTIFF=YES"])
    out_ds.SetGeoTransform(ds_dog1.GetGeoTransform())
    out_ds.SetProjection(ds_dog1.GetProjection())
    out_ds.GetRasterBand(1).WriteArray(SmoothedVariance)
    out_ds.GetRasterBand(1).SetNoDataValue(np.nan)
    out_ds.FlushCache()
    out_ds = None

    #now we can remove variance from memory:
    variance = None 

    print("Computing the actual normprod...")
    # Compute NormProd: (Im1 - smooth(Im1)) * (Im2 - smooth(Im2))
    normprod = dog1 * dog2

    print("Computing the sum of DoG1*DoG2 within small windows")
    # Apply a NaN-safe mean using generic_filter
    footprint = np.ones((window_size, window_size))  # Window for summation
    print("Starting generic_filter... about 10 mins for 11*11, 12 mins for 21*21.")
    #pdb.set_trace()
    summed_normprod = generic_filter(normprod, nan_safe_mean_filter, footprint=footprint, mode='constant', cval=np.nan)
    print("Finished generic_filter")
    
    print("Saving output...")
    # Save output
    out_ds = driver.Create(output_path, ds_dog1.RasterXSize, ds_dog1.RasterYSize, 1, gdal.GDT_Float32,
                           options=["COMPRESS=DEFLATE", "BIGTIFF=YES"])
    out_ds.SetGeoTransform(ds_dog1.GetGeoTransform())
    out_ds.SetProjection(ds_dog1.GetProjection())
    out_ds.GetRasterBand(1).WriteArray(summed_normprod)
    out_ds.GetRasterBand(1).SetNoDataValue(np.nan)
    out_ds.FlushCache()
    out_ds = None

    # Now also write out the divide-by-stdmean one....
    out_ds = driver.Create(output_path_std, ds_dog1.RasterXSize, ds_dog1.RasterYSize, 1, gdal.GDT_Float32,
                           options=["COMPRESS=DEFLATE", "BIGTIFF=YES"])
    out_ds.SetGeoTransform(ds_dog1.GetGeoTransform())
    out_ds.SetProjection(ds_dog1.GetProjection())
    # out_ds.GetRasterBand(1).WriteArray(summed_normprod/stdmean)
    out_ds.GetRasterBand(1).WriteArray(summed_normprod/SmoothedVariance)
    out_ds.GetRasterBand(1).SetNoDataValue(np.nan)
    out_ds.FlushCache()
    out_ds = None

    print("Freeing memory...")
    ds_dog1, ds_dog2, normprod, summed_normprod, SmoothedVariance = None, None, None, None, None
    print(f"✅ Saved NormProd (and STD) image: {output_path}")
    return output_path

def process_image_pair(base1, base2, localDataDir, outputEPSG, windows):
    """Processes an image pair to compute DoG and Normalized Product (NormProd)."""
    Date1, Date2 = base1[12:20], base2[12:20]
    
    # KEEPING compareFold unchanged
    compareFold = os.path.join(localDataDir, f"ISCE3_NormProd_EW_{Date1}_{Date2}")

    georeg1 = os.path.join(compareFold, f"georeg_1_{Date1}_EPSG{outputEPSG}.tif")
    georeg2 = os.path.join(compareFold, f"georeg_2_{Date2}_EPSG{outputEPSG}.tif")

    if not (os.path.exists(georeg1) and os.path.exists(georeg2)):
        print(f"❌ Skipping pair {base1}: missing input files.")
        return

    for sigma in windows:
        print(f"🚀 Processing DoG and NormProd with Gaussian smoothing sigma={sigma}")

        # Compute DoG images
        dog1_output = os.path.join(compareFold, f"DoG_{sigma}_EPSG{outputEPSG}_{Date1}.tif")
        dog2_output = os.path.join(compareFold, f"DoG_{sigma}_EPSG{outputEPSG}_{Date2}.tif")

        # DoGs weren't good, boxcar was much better.
        #compute_dog(georeg1, dog1_output, sigma)
        #compute_dog(georeg2, dog2_output, sigma)
        compute_boxcar_diff(georeg1, dog1_output, sigma)
        compute_boxcar_diff(georeg2, dog2_output, sigma)

        # Also need to compute the sigmas in here too...
        std1_output = os.path.join(compareFold, f"STD_{sigma}_EPSG{outputEPSG}_{Date1}.tif")
        std2_output = os.path.join(compareFold, f"STD_{sigma}_EPSG{outputEPSG}_{Date2}.tif")
        compute_local_std(georeg1, std1_output, sigma)
        compute_local_std(georeg2, std2_output, sigma)

        
        print("Happy with the filtering, and have made the STDs. Starting normprod...")

        # Compute Normalized Product (NormProd)
        normprod_output = os.path.join(compareFold, f"NormProd_{sigma}_EPSG{outputEPSG}_{Date1}_{Date2}.tif")
        normprod_std_output = os.path.join(compareFold, f"NormProdSmoVar_{sigma}_EPSG{outputEPSG}_{Date1}_{Date2}.tif")
        # normprod_std_output = os.path.join(compareFold, f"NormProdStd2_{sigma}_EPSG{outputEPSG}_{Date1}_{Date2}.tif")
        # normprod_std_output = os.path.join(compareFold, f"NormProdStdSq_{sigma}_EPSG{outputEPSG}_{Date1}_{Date2}.tif")
        std_output = os.path.join(compareFold, f"StdMean_{sigma}_EPSG{outputEPSG}_{Date1}_{Date2}.tif")
        smovar_output = os.path.join(compareFold, f"SmoothedVariance_{sigma}_EPSG{outputEPSG}_{Date1}_{Date2}.tif")
        compute_normprod(dog1_output, dog2_output, std1_output, std2_output, normprod_output, normprod_std_output, std_output, smovar_output, sigma)
        
        
    cleanup_intermediate_files(localDataDir+'/ISCE3_NormProd_EW_'+Date1+'_'+Date2)

        
# CLEANUP SCRIPT
def cleanup_intermediate_files(compareFold, keep_keywords=("SmoothedVariance", "NormProdSmoVar")):
    """
    Deletes .tif files in compareFold that do NOT contain any of the keep_keywords.
    """
    print("🧹 Starting cleanup of intermediate TIFFs...")
    all_files = os.listdir(compareFold)

    for fname in all_files:   
        if not fname.endswith(".tif"):
            continue

        if any(keyword in fname for keyword in keep_keywords):
            print(f"🛑 Keeping: {fname}")
            continue

        # If here, delete the file
        try:
            os.remove(os.path.join(compareFold, fname))
            print(f"🗑️ Deleted: {fname}")
        except Exception as e:
            print(f"⚠️ Could not delete {fname}: {e}")
