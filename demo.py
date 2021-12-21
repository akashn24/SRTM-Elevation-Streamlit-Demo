import numpy as np
from osgeo import gdal
import streamlit as st
from matplotlib import pyplot as plt
import pandas as pd
from mpl_toolkits import mplot3d
from matplotlib import cm
from matplotlib.colors import LightSource
from PIL import Image
import ee
import geemap.foliumap as geemap



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


def plottable_3d_info(df: pd.DataFrame):
    """
    Transform Pandas data into a format that's compatible with
    Matplotlib's surface and wireframe plotting.
    """
    index = df.index
    columns = df.columns
    
    y_label = df.index.to_series().apply(lambda x: np.round(x,2))
    x_label = df.columns.to_series().apply(lambda x: np.round(x,2))

    x, y = np.meshgrid(np.arange(len(columns)), np.arange(len(index)))
    z = np.array([[df[c][i] for c in columns] for i in index])
    
    xticks = dict(ticks=np.arange(len(columns)), labels=x_label)
    yticks = dict(ticks=np.arange(len(index)), labels=y_label)
    
    return x, y, z, xticks, yticks


def surface_plot(df_to_plot):
    ### Transform data from dataframe to Matplotlib friendly format.
    x, y, z, xticks, yticks = plottable_3d_info(df_to_plot)

    # Set up axes and put data on the surface.
    fig, axes = plt.subplots(subplot_kw=dict(projection='3d'))
    
    # Specifies colours of surface plot
    ls = LightSource(270, 45)
    rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')

    axes.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb, linewidth=0, antialiased=False, shade=False)

    axes.set_xlabel('Latitude')
    axes.set_ylabel('Longitude')
    axes.set_zlabel('Elevation')
    axes.set_zlim3d(bottom=0)

    # plt.xticks(fontsize=7, rotation = 45, **xticks)
    # plt.yticks(fontsize=7, **yticks)
    for t in axes.zaxis.get_major_ticks(): t.label.set_fontsize(7)

    plt.show()

    return fig
    

def plotting_for_app(fig, df, grid_size):
    """
    Creates section in a streamlit for the elevation map of an area in Malaysia
    based on the defined grid size
    """
    st.header(f"Elevation Plot {grid_size}x{grid_size}")
    st.markdown(
        f"""
    Elevation of area in West Malaysia constrained within latitude 3° to 4° and longitude 101° to 102, when area is split into a {grid_size}x{grid_size} grid.
    """
    )

    st.pyplot(fig)
    with st.expander(f"Data for {grid_size}x{grid_size}"):
        st.write(f"""
                    Dataframe of data which the {grid_size}x{grid_size} plot was based on
        """)
        st.dataframe(df)

def app(fig_1, df_1, fig_2, df_2):
    """
    Creates a simple streamlit displaying all the elevation maps
    and the google earth engine map
    """
    st.set_page_config(page_title="SRTM Data Elevation",layout='wide')
    st.title("NASA Elevation Data")

    st.header("Introduction")
    st.markdown(
        """
    This site demonstrates that the elevation of an area can be extracted from NASA's SRTM data found here: <https://www2.jpl.nasa.gov/srtm/>
    """
    )

    st.header("Area of Investigation")
    image = Image.open('Map_of_area_in_Malaysia.jpg')
    st.image(image, caption = "Slice of area of Malaysia which includes Kuala Lumpur and parts of Selangor, Pahang and Perak.\nThe area is constrained by latitude 3° to 4° and longitude 101° to 102.\nThe data which was extracted and visualised below, represents the elevation of this slice of area.")

    plotting_for_app(fig_1, df_1, 10)
    plotting_for_app(fig_2, df_2, 100)

    st.header("Google Earth Engine Map")
    st.markdown(
        """
    Map can be configured to add different layers of data. 
    Currently it includes the SRTM dataset and ECMWF heat map as well as the google earth map and terrain.
    """
    )

    # Create an interactive map
    Map = geemap.Map(plugin_Draw=True, Draw_export=False)
    # Add a basemap
    Map.add_basemap("TERRAIN")
    # Retrieve Earth Engine dataset for SRTM elevation data
    dem = ee.Image("USGS/SRTMGL1_003")
    # Retreive EE dataset for ERA-5-Land Hourly heat data
    heat = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_BY_HOUR").filter(ee.Filter.date('2020-06-01', '2020-07-01'))
    # Set visualization parameters for elevation
    vis_params_elev = {
        "min": 0,
        "max": 4000,
        "palette": ["006633", "E5FFCC", "662A00", "D8D8D8", "F5F5F5"],
    }
    # Set visualization parameters for heat
    vis_params_heat = {
        "bands": ['temperature_2m'],
        "min": 250.0,
        "max": 320.0,
        "palette": [
        "#000080","#0000D9","#4000FF","#8000FF","#0080FF","#00FFFF",
        "#00FF80","#80FF00","#DAFF00","#FFFF00","#FFF500","#FFDA00",
        "#FFB000","#FFA400","#FF4F00","#FF2500","#FF0A00","#FF00FF"],
    }
    # Add the Earth Engine images to the map
    Map.addLayer(dem, vis_params_elev, "SRTM DEM", True, 0.8)
    Map.addLayer(heat, vis_params_heat, "Air temperature [K] at 2m height", True, 0.8)
    # Add the colorbars to the map
    Map.add_colorbar(vis_params_elev["palette"], 0, 4000, caption="Elevation (m)")
    Map.add_colorbar(vis_params_heat["palette"], 250.0, 320.0, caption="Temperature (K)")
    # Render the map using streamlit
    Map.to_streamlit()
    


if __name__ == "__main__":
    # Simple examples of extracting the elevation data
    print('Mt Everest Elevation: %d' % read_raster_file(r'Elevation_Data_Files\N27E086.hgt', 27.9881, 86.9250))
    print('Kanchanjunga Elevation: %d' % read_raster_file(r'Elevation_Data_Files\N27E088.hgt', 27.7025, 88.1475))
    print('Kuala Lumpur Elevation: %d' % read_raster_file(r'Elevation_Data_Files\N03E101.hgt', 3.1390, 101.6869))

    elevation_data_df_10 = extract_elevation_of_area(r'Elevation_Data_Files\N03E101.hgt', 3, 101, 10)
    fig_10 = surface_plot(elevation_data_df_10)
    elevation_data_df_100 = extract_elevation_of_area(r'Elevation_Data_Files\N03E101.hgt', 3, 101, 100)
    fig_100 = surface_plot(elevation_data_df_100)

    app(fig_10, elevation_data_df_10, fig_100, elevation_data_df_100)


