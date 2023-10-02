"""



"""


from pyogrio import read_dataframe
from shapely.affinity import translate, scale
from shapely.validation import make_valid
from shapely.geometry import GeometryCollection, MultiPolygon
import re
import geopandas as gpd
import topojson as tp
from tqdm import tqdm
import aggregations
import pandas as pd

import warnings

warnings.filterwarnings("ignore")


def to_svg(shape, **kwargs):
    """SVG representation for iPython notebook"""
    svg_top = (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'xmlns:xlink="http://www.w3.org/1999/xlink" '
    )
    if shape.is_empty:
        return svg_top + "/>"
    else:
        # Establish SVG canvas that will fit all the data + small space
        xmin, ymin, xmax, ymax = shape.bounds
        if xmin == xmax and ymin == ymax:
            # This is a point; buffer using an arbitrary size
            xmin, ymin, xmax, ymax = shape.buffer(1).bounds
        else:
            # Expand bounds by a fraction of the data ranges
            expand = 0  # or 4%, same as R plots
            widest_part = max([xmax - xmin, ymax - ymin])
            expand_amount = widest_part * expand
            xmin -= expand_amount
            ymin -= expand_amount
            xmax += expand_amount
            ymax += expand_amount
        dx = xmax - xmin
        dy = ymax - ymin
        width = dx
        height = dy
        try:
            scale_factor = max([dx, dy]) / max([width, height])
        except ZeroDivisionError:
            scale_factor = 1.0
        view_box = "{} {} {} {}".format(xmin, ymin, dx, dy)
        transform = "matrix(1,0,0,-1,0,{})".format(ymax + ymin)
        return svg_top + (
            'width="{1}" height="{2}" viewBox="{0}" '
            'preserveAspectRatio="xMinYMin meet">'
            '<g transform="{3}">{4}</g></svg>'
        ).format(
            view_box,
            width,
            height,
            transform,
            shape.svg(scale_factor, **kwargs),
        )


def write_svg(shape, file):
    """
    Write the SVG representation of a shape to a file.

    Parameters:
    - shape (Shapely object): The geometric shape to convert to SVG.
    - file (str): The path to the file where the SVG will be written.
    """
    with open(file, "w") as f:
        svg = to_svg(shape)
        svg = re.sub('stroke-width="[\d\.]+"', 'stroke-width="0.25"', svg)
        svg = re.sub('opacity="[\d\.]+"', 'opacity="1"', svg)
        svg = re.sub('fill="#[0-9a-f]+"', 'fill="#D2EBD9"', svg)
        svg = re.sub('stroke="#[0-9a-f]+"', 'stroke="#A6A6A6"', svg)
        f.write(svg)


def scale_and_translate(shape, max_dim=200):
    """Scale and translate a shape to fit in a square of size max"""
    xmin, ymin, xmax, ymax = shape.bounds
    dx = xmax - xmin
    dy = ymax - ymin
    scale_factor = max_dim / max([dx, dy])
    shape_scaled = scale(shape, scale_factor, scale_factor)
    xmin, ymin, xmax, ymax = shape_scaled.bounds
    return translate(shape_scaled, -xmin, -ymin)


def remove_small_polygons(shape, cutoff=0.5):
    if shape.geom_type == "MultiPolygon":
        return MultiPolygon([g for g in shape.geoms if g.area > cutoff])
    return shape


def export_gpd_to_svg(df, path, simplify=True):
    """
    Export a GeoDataFrame to an SVG file.

    Parameters:
    - df (GeoDataFrame): The GeoDataFrame containing the geometries to export.
    - path (str): The path to the SVG file where data will be written.
    - simplify (bool, optional): Whether to simplify the shapes. Defaults to True.
    """

    num_shapes = df.shape[0]
    # Scale and translate will only work with native Shapely objects
    geom = GeometryCollection(df.geometry.to_list())
    geom = scale_and_translate(geom, max_dim=500)
    # Simplify shapes witout introducing artefacts for polygons sharing a border
    if simplify and num_shapes > 1:
        geom = tp.Topology(geom, prequantize=1e4, toposimplify=0.1).to_gdf()
        geom = GeometryCollection(geom.geometry.apply(remove_small_polygons).to_list())
    elif simplify:
        geom = geom.simplify(tolerance=0.1)
        geom = remove_small_polygons(geom)
    write_svg(geom, path)


def group_and_export_by(df, group_col, detail_level_col, simplify=True):
    grouped = df.groupby(group_col)
    for group, idx in list(grouped.groups.items()):
        df_group = df.loc[idx]
        if (df[detail_level_col] != "").sum() > 0:
            df_group = df_group.dissolve(by=detail_level_col)
            export_gpd_to_svg(
                df_group,
                f"./data/output_data/level_0/countries/{group}_{detail_level_col}.svg",
                simplify=simplify,
            )


def export_individual_country(df, name_column):
    """
    Export each individual country in a DataFrame to a separate SVG file.

    Parameters:
    - df (DataFrame): The DataFrame containing country data.
    - name_column (str): The column containing country ISO codes.
    """
    for iso in tqdm(df[name_column].unique()):
        df_iso = df[df[name_column] == iso]
        if iso == "GL":
            df_iso = apply_custom_crs(df_iso, degree=0)
        else:
            df_iso = apply_custom_crs(df_iso)
        export_gpd_to_svg(df_iso, f"./data/output_data/level_1/countries/{iso}.svg")


def export_aggregated_countries(df, name_column, iso_dic):
    """
    Export aggregated country data to an SVG file.

    Parameters:
    - df (DataFrame): The DataFrame containing country data.
    - name_column (str): The column containing country ISO codes.
    - iso_dic (dict): Dictionary containing the ISO codes of the aggregated countries.
    """

    if iso_dic["name"] != "World":
        df_iso = df[df[name_column].isin(iso_dic["iso_codes"])]
    else:
        df_iso = df[~df[name_column].isin(["AQ"])]
    export_gpd_to_svg(
        df_iso,
        f"./data/output_data/level_1/aggregations/{iso_dic['name'].lower().strip()}.svg",
    )


def apply_custom_crs(df, degree=-210):
    """
    Apply a custom coordinate reference system (CRS) to a DataFrame so that
    a country will not be splitted in half at the edge of the map.

    Parameters:
    - df (GeoDataFrame): The DataFrame to apply the CRS to.
    - degree (int, optional): The degree of rotation. Defaults to -210.

    Returns:
    - GeoDataFrame: The transformed DataFrame.
    """
    custom_mercator = f"+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0={degree} +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +wktext +no_defs"
    return df.to_crs(custom_mercator)


def change_crs(df, aggregation):
    """
    Change the CRS of a DataFrame based on aggregation details.

    Parameters:
    - df (GeoDataFrame): The DataFrame to change the CRS for.
    - aggregation (dict): Dictionary containing aggregation details.

    Returns:
    - GeoDataFrame: The transformed DataFrame.
    """
    if "america" in aggregation["name"].lower().strip():
        return apply_custom_crs(df, -10)
    return apply_custom_crs(df)


def is_admin_1_data(df):
    return "iso_a2" in df.columns


def is_admin_0_data(df):
    return "ISO_A2_EH" in df.columns


def check_iso_columns_present(df):
    assert is_admin_1_data(df) or is_admin_0_data(df)


def get_iso_col(df):
    return "ISO_A2_EH" if is_admin_0_data(df) else "iso_a2"


def fill_missing_iso_code(df):
    """
    Fill missing iso codes for admin level 1 data
    """
    if is_admin_1_data(df):
        df.at[44, "iso_a2"] = "SO"
        df.at[1626, "iso_a2"] = "CY"


def filter_out_irrelevant_type(df):
    """
    Filter out missing iso codes and territory types only representing small islands
    """
    if is_admin_1_data(df):
        return df[~((df.iso_a2 == "-1") | df.type_en.isin(["Overseas department"]))]
    return df[~((df.ISO_A2_EH == "-99") | df.TYPE.isin(["Dependency", "Lease"]))]


def keep_only_iso_codes_of_admin_0(df, name_column):
    """
    Filter out ISO codes which are not present in mapList.json.

    Only concerns small islands present in admin 1 dataset.
    """
    admin_0_iso_codes = pd.read_json("mapList.json").iloc[:, 0].unique()
    return df[df[name_column].isin(admin_0_iso_codes)]


def export_aggregation(countries, name_column):
    """
    Export aggregated country data based on predefined aggregations.

    Parameters:
    - countries (GeoDataFrame): The DataFrame containing country data.
    - name_column (str): The column containing country names or ISO codes.
    """
    for aggregation in tqdm(aggregations.aggregations):
        countries_adapted_crs = change_crs(countries, aggregation)
        export_aggregated_countries(countries_adapted_crs, name_column, aggregation)


if __name__ == "__main__":
    # file = "./data/raw_data/natural_earth/ne_10m_admin_0_countries_fra/ne_10m_admin_0_countries_fra.shp"
    file = "./data/raw_data/natural_earth/ne_10m_admin_1_states_provinces/ne_10m_admin_1_states_provinces.shp"
    df = read_dataframe(file)
    check_iso_columns_present(df)
    fill_missing_iso_code(df)
    iso_col = get_iso_col(df)
    df = filter_out_irrelevant_type(df)
    df = keep_only_iso_codes_of_admin_0(df, iso_col)

    export_individual_country(df, iso_col)
    export_aggregation(df, iso_col)
