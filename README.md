
# Mapping Crop Field with Geotagged Images
### Hanxi Li, CSISS, GMU

This project provides a Python package designed for converting geotagged images that contains crop information to crop maps by generating and analyzing crop field delineations using satellite imagery and machine learning. This project integrates geospatial processing tools such as Google Earth Engine, SAM (Segment Anything from Meta), and OpenStreetMap data to automate the pipeline for extracting and analyzing field boundaries and map the geotagged images to satellie view.

## Features

- **Metadata Extraction**: Extracts GPS and orientation data from image metadata (e.g., EXIF).
- **Sentinel-2 Image Processing**: Fetches Sentinel-2 composites from Google Earth Engine for a specified area and time.
- **Field Delineation**: Uses SAM (Segment Anything Model) to generate field boundary shapefiles from Sentinel-2 composites.
- **Mapping Crop Type**: Convert the labels from Geotagged Images to crop maps and save as geopackage file.

---

## Installation

1. Clone the repository:

2. Change Directory to the repository location:
   ```shell
   cd mapping_crop_field_with_geotagged_images
   ```
3. Create and activate the Conda environment:
   ```shell
   conda env create -f environment.yaml
   conda activate mapping_crop_field_env
   ```

---

## Usage

### Running the Pipeline

1. Open the workflow.ipynb
   
2. Use your own data or the example data to map geotagged images to crop map
---

## Folder Structure

```
MappingCropField/
├── MappingCropField/
│   ├── __init__.py
│   ├── main.py         
│   ├── boundary.py       
│   ├── metaextract.py    
│   ├── gee.py            
├── environment.yaml
├── README.md             
└── notebooks/ 
    ├── workflow.ipynb
```


## Acknowledgments

- SAM Model: Meta AI’s [Segment Anything Model](https://github.com/facebookresearch/segment-anything)
- Google Earth Engine for geospatial data.
- OpenStreetMap for road network data.

--- 
