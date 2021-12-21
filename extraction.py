import numpy as np
from osgeo import gdal
import pandas as pd

def read_raster_file(my_file, lat, lon):
    """
    Function to read in either HGT or TIFF files and to extract
    the elevation from the specified coordinates
    """
    data = gdal.Open(my_file)
    band1 = data.GetRasterBand(1)
    GT = data.GetGeoTransform()
    # GDAL's Affine Transformation (GetGeoTransform) 
    # https://gdal.org/tutorials/geotransforms_tut.html
    # GetGeoTransform translates latitude, longitude to pixel indices
    # GT[0] and GT[3] define the "origin": upper left pixel 
    x_pixel_size = GT[1]    #horizontal pixel size
    y_pixel_size = GT[5]    #vertical pixel size
    xP = int((lon - GT[0]) / x_pixel_size )
    yL = int((lat - GT[3]) / y_pixel_size )
    # without rotation, GT[2] and GT[4] are zero
    return (int( band1.ReadAsArray(xP,yL,1,1)))


def extract_elevation_of_area(file_name, lat_input, lon_input, no_of_points):
    # Uses above function to create an elevation map of a specified area
    latitude = np.linspace(lat_input, lat_input + 1, no_of_points)
    longitude = np.linspace(lon_input, lon_input + 1, no_of_points)

    df_elevation = pd.DataFrame(columns = latitude, index = longitude)

    for lon in df_elevation.index:
        for lat in df_elevation.columns:
            df_elevation.loc[lon][lat] = read_raster_file(file_name, lat, lon)
    
    # print(df_elevation)
    return df_elevation