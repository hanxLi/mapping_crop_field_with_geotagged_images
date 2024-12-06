"""
Earth Engine Image Processing and Geospatial Utilities
Generated with ChatGPT

This module provides tools for:
- Retrieving and processing Sentinel-2 image collections using Google Earth Engine (GEE).
- Exporting processed images to Google Drive and downloading them.
- Enhancing raster image contrast for visualization or further analysis.

Key Features:
-------------
1. Sentinel-2 Image Retrieval:
   - Fetch Sentinel-2 composites for specified areas and time ranges.
   - Preprocess images by applying cloud masking and scaling.

2. Earth Engine Export and Monitoring:
   - Export images from Earth Engine to Google Drive.
   - Monitor the progress of export tasks.

3. File Download and Raster Processing:
   - Download exported files from Google Drive using the Drive API.
   - Enhance the contrast of raster images by normalizing pixel values.

Dependencies:
-------------
- `os`: For file system operations.
- `time`: For monitoring task progress with delays.
- `ee`: For interacting with Google Earth Engine.
- `rasterio`: For reading and writing raster files.
- `numpy`: For numerical computations on image bands.
- `googleapiclient`: For interfacing with Google Drive.
- `google.oauth2`: For authentication using service account credentials.

Functions:
----------
- `s2_composite(doi, sample_aoi, band_list)`:
    Retrieves Sentinel-2 image collections for a given area and date range.
    
- `export_eeImg(ee_image, export_folder, export_filename, scale=10)`:
    Exports Earth Engine images to Google Drive.

- `monitor_task(task, verbose=False)`:
    Monitors the progress of Earth Engine export tasks.

- `download_from_drive(file_name, folder_id, save_path, credentials_file, verbose=False)`:
    Downloads a file from Google Drive to a local directory.

- `enhance_raster_contrast(file_path, verbose=False)`:
    Enhances the contrast of raster images by stretching pixel values.

Notes:
------
Some functions in this module are inspired or adapted from:
https://github.com/chchang1990/SAM_field_delineation
"""
import os
import io
import time

import ee
import rasterio
import numpy as np

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.service_account import Credentials

def s2_composite(doi, sample_aoi, band_list):
    """
    Retrieve Sentinel-2 image collection of defined bands for a given acquisition time within an AOI.

    Parameters:
    -----------
    doi : str
        Date of interest (start date) in 'YYYY-MM-DD' format.
    sample_aoi : ee.Geometry
        Area of interest as an Earth Engine geometry.
    band_list : list of str
        List of Sentinel-2 bands to include in the composite.

    Returns:
    --------
    ee.ImageCollection
        A preprocessed Sentinel-2 image collection.

    Notes:
    -------
    This portion of the code is borrowed from https://github.com/chchang1990/SAM_field_delineation
    """
    # A mapping function for cloud masking and value scaling
    def maskS2clouds(image):
        qa = image.select('QA60')

        # QA has 12 bits from bit-0 to bit-11
        # Bits 10 and 11 are clouds and cirrus, respectively.
        cloudBitMask = 1 << 10  # Push "1" 10 spaces to the left (010000000000)
        cirrusBitMask = 1 << 11  # Push "1" 11 spaces to the left (100000000000)

        # "bitwiseAnd" compare the QA and "cloudBitMask" and "cirrusBitMask"
        # then return True if (1) bit-10 of QA and "cloudBitMask" do not match (no cloud)
        #                     (2) bit-11 of QA and "cirrusBitMask" do not match (no cirrus cloud)
        #
        # Both flags should be set to zero, indicating clear conditions.
        # (a pixel that is neither cloud nor cirrus cloud will be retained)
        mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
        #
        # Mask out the cloud/cirrus-cloud pixels 
        # and scale the data by 10000 (the scale factor of the data)
        return image.updateMask(mask).divide(10000).copyProperties(image, ['system:time_start'])

    # A mapping function to scale the composited images to 0-255 range
    def rgb_uint8(image):
        return image.multiply(255).uint8().copyProperties(image, ['system:time_start'])

    # A mapping function to clip the image collection to the defined AOI
    def imgcol_clip(image):
        return image.clip(sample_aoi)


    doi_end = ee.Date(doi).advance(30,'day').format('YYYY-MM-dd')

    # Retrieve the selected bands of S2 image collection
    s2_composite_ImgCol = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterDate(doi, doi_end)
        ##Pre-filter to get less cloudy granules.
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',20))
        .filterBounds(sample_aoi)
    ).map(maskS2clouds).select(band_list).map(rgb_uint8).map(imgcol_clip)

    return s2_composite_ImgCol


def export_eeImg(ee_image, export_folder, export_filename, scale=10):
    """
    Exports an Earth Engine image to Google Drive.

    Parameters:
    -----------
    ee_image : ee.Image
        Earth Engine image to export.
    export_folder : str
        Google Drive folder name for export.
    export_filename : str
        Filename for the exported image.
    scale : int
        Resolution of the exported image in meters.

    Returns:
    --------
    ee.batch.Task
        The export task object.
    
    Notes:
    --------
    This portion of the code is borrowed from https://github.com/chchang1990/SAM_field_delineation
    """
    task = ee.batch.Export.image.toDrive(**{
        'image': ee_image,
        'description': export_filename,
        'folder': export_folder,
        'scale': scale,
        'region': ee_image.geometry().getInfo()['coordinates']
    })
    task.start()
    return task

def monitor_task(task, verbose=False):
    """
    Monitors the status of an Earth Engine export task.

    Parameters:
    -----------
    task : ee.batch.Task
        The Earth Engine task to monitor.
    verbose : bool
        If True, prints task status updates.

    Returns:
    --------
    bool
        True if the task completes successfully, False otherwise.

    Notes:
    --------
    This portion of the code is modified by ChatGPT
    """
    if verbose:
        print("Monitoring export task...")
    while True:
        status = task.status()
        state = status['state']
        if state == 'COMPLETED':
            if verbose:
                print("Export completed successfully.")
            return True
        elif state == 'FAILED':
            print(f"Export failed: {status['error_message']}")
            return False
        else:
            time.sleep(10)

def download_from_drive(file_name, folder_id, save_path, credentials_file, verbose=False):
    """
    Downloads a file from Google Drive to a local path using its folder ID.

    Parameters:
    -----------
    file_name : str
        Name of the file to download.
    folder_id : str
        Google Drive folder ID containing the file.
    save_path : str
        Local path to save the downloaded file.
    credentials_file : str
        Path to the service account credentials JSON file.
    verbose : bool
        If True, prints download progress.

    Returns:
    --------
    None

    Notes:
    --------
    This portion of the code is modified by ChatGPT
    """

    creds = Credentials.from_service_account_file(credentials_file, 
                                                  scopes=['https://www.googleapis.com/auth/drive'])
    drive_service = build('drive', 'v3', credentials=creds)

    # Search for the file using folder ID
    query = f"name = '{file_name}' and '{folder_id}' in parents"
    response = drive_service.files().list(q=query,
                                          spaces='drive', fields='files(id, name)').execute()
    files = response.get('files', [])

    if not files:
        print(f"File {file_name} not found in Google Drive folder with ID {folder_id}.")
        return

    file_id = files[0]['id']
    request = drive_service.files().get_media(fileId=file_id)

    # Download the file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure the directory exists
    with io.FileIO(save_path, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if verbose:
                print(f"Download progress: {int(status.progress() * 100)}%")
    if verbose:
        print(f"File downloaded to {save_path}")

def enhance_raster_contrast(file_path, verbose=False):
    """
    Enhances the contrast of a raster image by stretching each band's values to 0-255.

    Parameters:
    -----------
    file_path : str
        Path to the input raster file.
    verbose : bool
        If True, prints progress information.

    Returns:
    --------
    None
    """
    try:
        with rasterio.open(file_path) as src:
            # Read raster metadata
            meta = src.meta

            # Prepare for enhanced contrast
            enhanced_bands = []
            for i in range(1, src.count + 1):  # Process each band
                band = src.read(i)
                min_val, max_val = np.percentile(band, (2, 98))  # Stretch between 2nd and 98th percentiles
                stretched_band = np.clip((band - min_val) / (max_val - min_val) * 255, 0, 255).astype(np.uint8)
                enhanced_bands.append(stretched_band)

            # Update metadata for enhanced raster
            meta.update(dtype='uint8')

            # Write enhanced bands back to the same file
            with rasterio.open(file_path, 'w', **meta) as dst:
                for i, band in enumerate(enhanced_bands, start=1):
                    dst.write(band, i)
            if verbose:
                print(f"Contrast enhanced and saved in: {file_path}")

    except Exception as e:
        print(f"Error processing raster {file_path}: {e}")
