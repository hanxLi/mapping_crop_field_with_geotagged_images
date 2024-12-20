"""
Field Delineation and Road Network Analysis

This module provides utilities for analyzing geographic raster data, extracting road networks, 
and delineating field boundaries using various geospatial libraries.

Generated by ChatGPT

Key Functionality:
------------------
1. Generate a bounding box around a specific coordinate.
2. Segment raster images into polygons using SAM (Segment Anything Model).
3. Extract road networks from OpenStreetMap data and update field delineations 
   by removing polygons that touch roads.

Dependencies:
-------------
- os: For file and directory operations.
- rasterio: For reading and manipulating GeoTIFF raster data.
- osmnx: For querying and working with OpenStreetMap road networks.
- geopandas: For working with geospatial vector data.
- shapely: For geometric operations like buffering and intersections.
- pyproj: For CRS transformations.
- matplotlib.pyplot: For visualization.
- samgeo: For running the SAM segmentation model.

Functions:
----------
- generate_bounding_box(lat, lon, radius_km):
    Generates a bounding box around a given point.

- segment_with_sam(input_image_path, output_folder, sam_model):
    Segments raster images into polygons using SAM and outputs results as a GeoPackage.

- extract_roads_and_update_delineations(geotiff_path, field_delineation_path, min_area_threshold, network_type, visualize):
    Extracts road networks from OpenStreetMap data, applies buffers, updates field delineations, 
    and optionally visualizes the results.

"""
import os

import rasterio
import osmnx as ox
import geopandas as gpd
from rasterio.plot import show
from shapely.geometry import Point, LineString, box
from shapely.ops import transform
from pyproj import Transformer
import matplotlib.pyplot as plt

def generate_bounding_box(lat, lon, radius_km):
    """
    Generate a bounding box around a coordinate with a specified radius.

    Parameters:
    -----------
    lat : float
        Latitude of the center point.
    lon : float
        Longitude of the center point.
    radius_km : float
        Radius in kilometers for the bounding area.

    Returns:
    --------
    tuple
        Bounding box as (min_lon, min_lat, max_lon, max_lat).
    
    Notes:
    --------
    The CRS transformation is inspired and partially modified by ChatGPT
    """
    radius_m = radius_km * 1000

    point = Point(lon, lat)

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    reverse_transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    projected_point = transform(transformer.transform, point)

    buffered_area = projected_point.buffer(radius_m)
    min_x, min_y, max_x, max_y = buffered_area.bounds

    min_lon, min_lat = transform(reverse_transformer.transform, Point(min_x, min_y)).coords[0]
    max_lon, max_lat = transform(reverse_transformer.transform, Point(max_x, max_y)).coords[0]

    return min_lon, min_lat, max_lon, max_lat

def segment_with_sam(input_image_path, output_folder, sam_model):
    """
    Segments the input image into polygons using SAM and saves the output as a GeoPackage.

    Parameters:
    -----------
    input_image_path : str
        Path to the input raster image.
    output_folder : str
        Path to the folder for saving the segmented outputs.
    sam_model : SamGeo
        Initialized SAM model instance.

    Returns:
    --------
    str
        Path to the generated GeoPackage with segmented polygons.

    Notes:
    --------
    This portion of the code is inspired by https://github.com/chchang1990/SAM_field_delineation
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    base_name = os.path.basename(input_image_path)[:-4]
    output_segment_path = os.path.join(output_folder, f"{base_name}_delineation.tif")
    output_segment_vector_path = os.path.join(output_folder, f"{base_name}_delineation.gpkg")

    try:
        sam_model.generate(
            input_image_path, 
            output_segment_path, 
            batch=True, 
            foreground=True, 
            erosion_kernel=(3, 3), 
            mask_multiplier=255
        )

        sam_model.tiff_to_gpkg(output_segment_path, output_segment_vector_path, simplify_tolerance=None)

        if os.path.exists(output_segment_path):
            os.remove(output_segment_path)

    except Exception as e:
        print(f"Error during segmentation: {e}")
        return None
    return output_segment_vector_path

def extract_roads_and_update_delineations(geotiff_path, field_delineation_path, min_area_threshold=5000, network_type='drive', visualize=False):
    """
    Extracts road networks within the bounds of a GeoTIFF, or falls back to a point buffer 
    if no road network is found. Updates field delineations accordingly.
    """
    with rasterio.open(geotiff_path) as src:
        bounds = src.bounds
        raster_crs = src.crs

    transformer_to_wgs84 = Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True)
    min_x, min_y, max_x, max_y = bounds
    left, bottom = transformer_to_wgs84.transform(min_x, min_y)
    right, top = transformer_to_wgs84.transform(max_x, max_y)
    bbox = (left, bottom, right, top)

    try:
        G = ox.graph.graph_from_bbox(
            bbox=bbox,
            network_type=network_type,
            simplify=True,
            retain_all=False
        )
        _, gdf_edges = ox.graph_to_gdfs(G)
    except ValueError:
        print("No roads found. Falling back to point buffer.")
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        center_point = Point(center_x, center_y)
        gdf_edges = gpd.GeoDataFrame(
            geometry=[center_point.buffer(200)],
            crs=raster_crs
        )

    if gdf_edges.crs != raster_crs:
        gdf_edges = gdf_edges.to_crs(raster_crs)

    raster_bounds_geom = box(*bounds)
    gdf_edges_clipped = gdf_edges[gdf_edges.intersects(raster_bounds_geom)]

    delineations = gpd.read_file(field_delineation_path)
    if delineations.crs != raster_crs:
        delineations = delineations.to_crs(raster_crs)

    if not gdf_edges_clipped.empty:
        delineations_no_touch = delineations[~delineations.geometry.intersects(gdf_edges_clipped.unary_union)]
        buffered_roads = gdf_edges_clipped.buffer(200)
        filtered_delineations = delineations_no_touch[delineations_no_touch.geometry.intersects(buffered_roads.unary_union)].copy()
    else:
        print("Using point-based buffer for filtering delineations.")
        buffered_point = center_point.buffer(200)
        filtered_delineations = delineations[delineations.geometry.intersects(buffered_point)].copy()

    filtered_delineations["area"] = filtered_delineations.geometry.area
    filtered_delineations = filtered_delineations[filtered_delineations["area"] >= min_area_threshold]

    if visualize:
        plt.figure(figsize=(12, 12))
        plt.title("Road Network and Filtered Field Delineations")
        with rasterio.open(geotiff_path) as src:
            img = src.read(1)
            extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
            plt.imshow(img, extent=extent, cmap='gray', origin='upper')
        gdf_edges_clipped.plot(ax=plt.gca(), color='red', linewidth=1, label='Road Network')
        filtered_delineations.plot(ax=plt.gca(), edgecolor='blue', facecolor='none', linewidth=1, label='Filtered Field Delineations')
        plt.axis("off")
        plt.show()

    filtered_delineations.to_file(field_delineation_path, driver="GPKG")
    return gdf_edges_clipped


# def extract_roads_and_update_delineations(geotiff_path, field_delineation_path, min_area_threshold=1000, network_type='drive' ,visualize=False):
#     """
#     Extracts road networks within the bounds of a GeoTIFF, updates field delineations by removing 
#     shapes that touch the road network, applies a buffer around the road network, and retains only 
#     delineations that touch the buffer.

#     Parameters:
#     -----------
#     geotiff_path : str
#         Path to the input GeoTIFF file.
#     field_delineation_path : str
#         Path to the GeoPackage file containing field delineations.
#     min_area_threshold : float
#         Minimum area threshold for polygons (in CRS units, e.g., square meters).
#     network_type : str
#         Type of road network to retrieve (e.g., 'drive', 'walk', etc.).
#     visualize : bool
#         If True, visualizes the road network and filtered delineations.

#     Returns:
#     --------
#     GeoDataFrame
#         GeoDataFrame containing the clipped road network.

#     Notes:
#     --------
#     The OSM download part is inspired by ChatGPT (have to fix its output for like 3 hrs lol)
#     """
#     with rasterio.open(geotiff_path) as src:
#         bounds = src.bounds
#         raster_crs = src.crs

#     transformer_to_wgs84 = Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True)
#     min_x, min_y, max_x, max_y = bounds
#     left, bottom = transformer_to_wgs84.transform(min_x, min_y)
#     right, top = transformer_to_wgs84.transform(max_x, max_y)
#     bbox = (left, bottom, right, top)

#     G = ox.graph.graph_from_bbox(
#         bbox=bbox,
#         network_type=network_type,
#         simplify=True,
#         retain_all=False
#     )
#     _, gdf_edges = ox.graph_to_gdfs(G)

#     if gdf_edges.crs != raster_crs:
#         gdf_edges = gdf_edges.to_crs(raster_crs)

#     raster_bounds_geom = box(*bounds)
#     gdf_edges_clipped = gdf_edges[gdf_edges.intersects(raster_bounds_geom)]

#     if gdf_edges_clipped.empty:
#         raise ValueError("No roads found within the given GeoTIFF bounds.")

#     delineations = gpd.read_file(field_delineation_path)
#     if delineations.crs != raster_crs:
#         delineations = delineations.to_crs(raster_crs)

#     delineations_no_touch = delineations[~delineations.geometry.intersects(gdf_edges_clipped.union_all())].copy()

#     buffered_roads = gdf_edges_clipped.buffer(200)

#     filtered_delineations = delineations_no_touch[delineations_no_touch.geometry.intersects(buffered_roads.union_all())].copy()

#     filtered_delineations["area"] = filtered_delineations.geometry.area
#     filtered_delineations = filtered_delineations[filtered_delineations["area"] >= min_area_threshold]

#     if visualize:
#         plt.figure(figsize=(12, 12))
#         plt.title("Road Network and Filtered Field Delineations")

#         with rasterio.open(geotiff_path) as src:
#             img = src.read(1)
#             extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
#             plt.imshow(img, extent=extent, cmap='gray', origin='upper')

#         gdf_edges_clipped.plot(ax=plt.gca(), color='red', linewidth=1, label='Road Network')

#         filtered_delineations.plot(ax=plt.gca(), edgecolor='blue', facecolor='none', linewidth=1, label='Filtered Field Delineations')

#         plt.axis("off")
#         plt.show()

#     filtered_delineations.to_file(field_delineation_path, driver="GPKG")
#     return gdf_edges_clipped
