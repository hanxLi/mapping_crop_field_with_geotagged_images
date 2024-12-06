"""
MappingCropField Package

This package provides tools for:
- Extracting metadata from images.
- Generating Sentinel-2 composites and field delineations.
- Processing field delineations using SAM and OpenStreetMap data.
- Performing geospatial operations like road extraction and raster contrast enhancement.

Modules:
--------
- `metaextract`: Tools for extracting GPS metadata from images.
- `boundary`: Utilities for field delineation and road network extraction.
- `gee`: Functions for interacting with Google Earth Engine and raster processing.
- `main`: High-level pipeline functions for end-to-end processing.
"""
from .main import extract_metadata, generate_delineation, assign_crop_field, display_assigned_field

from .boundary import extract_roads_and_update_delineations
from .gee import s2_composite, enhance_raster_contrast

# Define what is available when using `from mappingCropField import *`
__all__ = [
    "generate_delineation",
    "assign_crop_field",
    "display_assigned_field",
    "extract_metadata",
    "extract_roads_and_update_delineations",
    "s2_composite",
    "enhance_raster_contrast",
]
