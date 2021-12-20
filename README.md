# SRTM-Elevation-Streamlit-Demo
A simple streamlit demo, showing how to extract the elevation data from an SRTM file, and also to show how the entire SRTM dataset looks like on a global scale.

The main SRTM file reading function was created by [Nicholas-Fong](https://github.com/nicholas-fong/SRTM-GeoTIFF), so if you would like to know more about it then feel free to visit his repository. This function can be used to read files of type: `GeoTIFF`, `DTED`, `HGT`, `BIL`

For demonstration purposes I personally investigated an area in Malaysia constrained by latitude 3° and 4°, and longitude 101° and 102°. The file for the data of these coordinates are contained in: `N03E101.hgt`. I have also included files `N27E088.hgt` and `N27E086.hgt` to demonstrate the extraction for the elevation of Kanchanjunga and Mt. Everest respectively. For more information on the naming convention of files, it is explained well on the [SRTM FAQ](https://www2.jpl.nasa.gov/srtm/faq.html#wrapper_bottom) website.
