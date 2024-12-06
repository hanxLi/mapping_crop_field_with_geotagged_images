"""
Mapping and Analysis Pipeline for Crop Fields

This module contains functions for extracting metadata, generating Sentinel-2 composites, 
performing segmentation with SAM, and assigning crop fields based on imagery metadata.

Generated with ChatGPT

Key Functions:
--------------
- extract_metadata(folder_path): Extract GPS metadata from image files and return a GeoDataFrame.
- generate_delineation(df, color_composite, save_path, credentials_file_path, folder_id, radius_km=1.5, visualize=False): 
  Generate Sentinel composites and field delineations using SAM and Google Earth Engine.
- assign_crop_field(df, root_dir, batch_process=False, field_path=None): 
  Assign crop field geometries to imagery metadata.
- display_assigned_field(df, color_composite, save_path, credentials_file_path, folder_id, batch_process=False): 
  Combine delineation generation and crop field assignment into a single pipeline for visualization or batch processing.

Dependencies:
-------------
- pandas: For handling tabular data.
- geopandas: For spatial data manipulation.
- rasterio: For raster processing.
- matplotlib: For visualization.
- shapely: For geometric operations.
- pyproj: For coordinate transformations.
- geemap & ee: For Google Earth Engine operations.
- samgeo: For SAM-based segmentation.
- metaextract, boundary, gee: Custom modules for metadata extraction, 
  boundary processing, and GEE integration.
"""
import os
import math

import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import Point, LineString, box
import matplotlib.pyplot as plt
from pyproj import Transformer
import geemap
import ee
from samgeo import SamGeo

from .metaextract import get_gps_metadata_from_file
from .boundary import segment_with_sam, generate_bounding_box, extract_roads_and_update_delineations
from .gee import export_eeImg, s2_composite, monitor_task, download_from_drive, enhance_raster_contrast


def extract_metadata(folder_path):
    """
    Extracts GPS metadata from images in the given folder and returns a GeoDataFrame.

    Parameters:
    -----------
    folder_path (str): 
        Path to the folder containing image files.

    Returns:
    --------
    GeoDataFrame: 
        A GeoDataFrame containing extracted metadata such as filename, image path, altitude, 
        direction, timestamp, and geometry (Point with latitude and longitude).
    """
    metadata_list = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(os.path.join(root, file), start=os.getcwd())


            if file.lower().endswith(('.jpg', '.jpeg')):
                try:
                    gps_data = get_gps_metadata_from_file(file_path)

                    latitude = gps_data.get('Latitude')
                    longitude = gps_data.get('Longitude')
                    altitude = gps_data.get('Altitude (meters)', None)
                    direction = gps_data.get('Image Direction (degrees)', None)
                    date_taken = gps_data.get('Date Taken')

                    # Only include images with valid latitude and longitude
                    if latitude is not None and longitude is not None:
                        # Create metadata entry
                        metadata_entry = {
                            "filename": file,
                            "img_path": rel_path,
                            "altitude": altitude,
                            "direction": direction,
                            "time": date_taken,
                            "geometry": Point(longitude, latitude),  
                        }
                        metadata_list.append(metadata_entry)
                except Exception as e:
                    # Skip files that don't have valid metadata or cause errors
                    print(f"Skipping file {file}: {e}")

    metadata_df = pd.DataFrame(metadata_list)

    if not metadata_df.empty:
        metadata_gdf = gpd.GeoDataFrame(metadata_df, geometry="geometry", crs="EPSG:4326")
    else:
        metadata_gdf = gpd.GeoDataFrame(columns=["filename", "img_path", "altitude", "direction", "time", "geometry"])
    return metadata_gdf

def generate_delineation(df, color_composite, save_path, credentials_file_path, folder_id,radius_km=1.5, visualize=False):
    """
    Processes input data to generate Sentinel composites and field delineations with SAM.

    Parameters:
    -----------
    df (GeoDataFrame): 
        Input GeoDataFrame with metadata and geometry.
    color_composite (str): 
        Type of composite to generate ('true_color' or 'false_color').
    save_path (str): 
        Path that contains the saved files.
    credentials_file_path (str): 
        Path to the credentials JSON file for accessing Google Drive API.
    folder_id (str): 
        Folder ID for downloading GEE outputs on Google Drive.
    radius_km (float): 
        Radius for generating bounding boxes around each input point (in kilometers).
    visualize (bool): 
        If True, displays the map with Sentinel images and delineations on an interactive map.

    Returns:
    --------
    GeoDataFrame: 
        GeoDataFrame with original data and additional processed information, including 
        raster paths and delineation paths.
    """
    processed_data = []

    for i in range(len(df)):
        lat = df.loc[i].geometry.y
        lon = df.loc[i].geometry.x
        bbox = generate_bounding_box(lat, lon, radius_km)
        aoi_w, aoi_s, aoi_e, aoi_n = bbox
        aoi = ee.Geometry.BBox(aoi_w, aoi_s, aoi_e, aoi_n)

        time_taken = df.loc[i]["time"].split(" ")[0]
        doi = time_taken.split(":")[0] + "-" + time_taken.split(":")[1] + "-01"

        band_combinations = {
            'true_color': ['B4', 'B3', 'B2'],
            'false_color': ['B8', 'B4', 'B3']
        }
        selected_bands = band_combinations[color_composite]

        s2_img = s2_composite(doi, aoi, selected_bands).first()

        export_folder = "sen2imgs"
        file_name = df.loc[i].filename.split(".")[0] + f"_s2_{color_composite}" + ".tif"
        local_save_path = os.path.join(save_path, file_name)
        output_folder = os.path.join(save_path, "out")
        input_image_path = os.path.join(save_path, file_name)

        task = export_eeImg(s2_img, export_folder, file_name.split(".")[0], scale=10)

        if monitor_task(task):
            download_from_drive(
                file_name=file_name,
                folder_id=folder_id,
                save_path=local_save_path,
                credentials_file=credentials_file_path
            )

        enhance_raster_contrast(local_save_path)

        # Initialize SAM
        sam = SamGeo(
            model_type="vit_h",
            checkpoint="sam_vit_h_4b8939.pth",
            sam_kwargs=None,
        )

        # Segment using SAM
        gpk_segment_path = segment_with_sam(input_image_path, output_folder, sam)

        rel_raster_path = os.path.relpath(local_save_path, start=save_path)
        rel_delineation_path = os.path.relpath(gpk_segment_path, start=save_path)

        processed_entry = df.loc[i].to_dict()
        processed_entry.update({
            "raster_path": rel_raster_path,
            "delineation_path": rel_delineation_path
        })
        processed_data.append(processed_entry)

    processed_gdf = gpd.GeoDataFrame(processed_data, geometry="geometry", crs=df.crs)
    column_order = ["filename", "img_path", "altitude", "direction",
                     "time", "raster_path", "delineation_path", "crop_type", "geometry"]

    processed_gdf = processed_gdf[column_order]

    if visualize:
        Map = geemap.Map(center=((aoi_n + aoi_s) / 2, (aoi_w + aoi_e) / 2), zoom=15)
        Map.add_vector(gpk_segment_path, layer_name="SAM2 Segments")
        Map.addLayer(s2_img, 
                     {'min': 0, 'max': 255, 'bands': selected_bands, 'gamma': 1.7}, 
                     name="Sentinel 2 IMG")
        Map.addLayerControl()
        return Map

    return processed_gdf

def assign_crop_field(df, root_dir, batch_process=False, field_path=None):
    """
    Processes field delineations and assigns crop field geometries to the input photo.

    Parameters:
    -----------
    df (GeoDataFrame): 
        Input GeoDataFrame containing image metadata and geometry.
    root_dir (str): 
        Root directory containing raster and delineation files.
    batch_process (bool): 
        If True, skips visualization and saves results to a GeoPackage file.
    field_path (str): 
        Directory path to save the resulting GeoDataFrame when batch_process=True.

    Returns:
    --------
    GeoDataFrame: 
        Processed GeoDataFrame with selected polygon geometries and additional metadata.
    """
    results = []

    for i in range(len(df)):
        angle = df.loc[i].direction
        angle_rad = math.radians(90 - angle)
        raster_path = os.path.join(root_dir, df.loc[i].raster_path)
        delineation_path = os.path.join(root_dir, df.loc[i].delineation_path)
        lat = df.loc[i].geometry.y
        lon = df.loc[i].geometry.x

        road_df = extract_roads_and_update_delineations(raster_path, delineation_path)

        with rasterio.open(raster_path) as raster:
            bounds = raster.bounds
            img = raster.read(1)
            raster_crs = raster.crs

            transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
            coord_transformed = transformer.transform(lon, lat)

            start_point = Point(coord_transformed)

            distance = 1e6  # Arbitrary large distance
            end_x = start_point.x + distance * math.cos(angle_rad)
            end_y = start_point.y + distance * math.sin(angle_rad)
            extended_line = LineString([start_point, (end_x, end_y)])

            raster_bounds = box(*bounds)
            clipped_line = extended_line.intersection(raster_bounds)

            delineation = gpd.read_file(delineation_path)
            delineation = delineation.to_crs(raster_crs)

            delineation["intersects"] = delineation.intersects(clipped_line)
            intersections = delineation[delineation["intersects"]]

            selected_shape = None
            if not intersections.empty:
                intersections["distance"] = intersections.geometry.apply(lambda geom: geom.distance(start_point))
                selected_shape = intersections.sort_values("distance").iloc[0].geometry

                results.append({
                    "filename": df.loc[i].filename,
                    "time": df.loc[i].time,
                    "crop_type": df.loc[i].crop_type,
                    "geometry": selected_shape
                })

            if not batch_process:
                _, ax = plt.subplots(figsize=(12, 12))
                extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
                ax.imshow(img, extent=extent, cmap='gray', origin='upper')

                delineation.plot(ax=ax, edgecolor="gray", facecolor="none", 
                                 linewidth=1, label="All Shapes")
                if selected_shape is not None:
                    gpd.GeoSeries([selected_shape], crs=raster_crs).plot(ax=ax,
                                                                         edgecolor="green",
                                                                         facecolor="none",
                                                                         linewidth=2,
                                                                         label="Selected Field")
                gpd.GeoSeries([clipped_line], crs=raster_crs).plot(ax=ax,
                                                                   color="blue",
                                                                   linewidth=2,
                                                                   label="Direction Line")
                gpd.GeoSeries([start_point], crs=raster_crs).plot(ax=ax,
                                                                  color="red",
                                                                  markersize=50,
                                                                  label="POV")
                road_df.plot(ax=ax, color='orange', linewidth=1, label="Road")
                ax.set_title("Field Delineation with Target Highlighted")
                plt.axis("off")
                plt.show()

    processed_gdf = gpd.GeoDataFrame(results, geometry="geometry", crs=raster_crs)

    if batch_process and field_path:
        processed_gdf.to_file(os.path.join(field_path, "labeled_delineation.gpkg"), driver="GPKG")

    return processed_gdf


def display_assigned_field(df, color_composite, save_path, credentials_file_path, folder_id, batch_process=False):
    """
    Generates field delineations, assigns crop fields, 
    and optionally saves results using previous functions.

    Parameters:
    -----------
    df (GeoDataFrame): 
        Input GeoDataFrame with metadata and geometry.
    color_composite (str): 
        Type of composite to generate ('true_color' or 'false_color').
    save_path (str): 
        Directory path to save the processed files and results.
    credentials_file_path (str): 
        Path to the credentials JSON file for accessing Google Drive API.
    folder_id (str): 
        Folder ID for storing outputs on Google Drive.
    batch_process (bool): 
        If True, skips visualization and saves results to a GeoPackage file.

    Returns:
    --------
    GeoDataFrame: 
        Processed GeoDataFrame containing field delineation and crop field assignments.
    """
    filtered_df = generate_delineation(df,
                                       color_composite,
                                       save_path,
                                       credentials_file_path,
                                       folder_id,
                                       visualize=False)
    gdf = assign_crop_field(filtered_df,
                            root_dir = save_path,
                            batch_process=batch_process,
                            field_path = save_path)
    return gdf
