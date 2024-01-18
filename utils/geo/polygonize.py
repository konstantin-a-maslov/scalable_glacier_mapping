import rasterio
import rasterio.features
import geopandas
import argparse


def polygonize(input_path, output_path):
    with rasterio.open(input_path, "r") as src:
        labels = src.read(1)
        trans = src.transform
        crs = src.crs

    polygons = [
        {"properties": {"id": label}, "geometry": geom}
        for geom, label in rasterio.features.shapes(labels, transform=trans)
        if label != 0
    ]
    
    dataframe = geopandas.GeoDataFrame.from_features(polygons, crs=crs)
    dataframe.to_file(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to input .tif file")
    parser.add_argument("output_path", help="Path to output .shp file")
    args = parser.parse_args()

    polygonize(args.input_path, args.output_path)
