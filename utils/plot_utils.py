# Geospatial data handling
import matplotlib as mpl

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from pathlib import Path

def set_matplotlib_params():
    # 기본 글꼴을 serif로 설정
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern serif font 사용
    mpl.rcParams['mathtext.rm'] = 'serif'
    mpl.rcParams['mathtext.it'] = 'serif:italic'
    mpl.rcParams['mathtext.bf'] = 'serif:bold'
    
    custom_color_dict = {
        'custom_navy': (26/255, 61/255, 102/255)
    }
    return custom_color_dict

def plot_korea(figure_path, ax, x1, x2, y1, y2, zorder_base=0, boundary_lw=0.7):

    # Add South Korea's borders, cities, and provincial boundaries
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='black', zorder=zorder_base+1)
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', zorder=zorder_base+1)
    #ax.add_feature(cfeature.LAKES, alpha=0.5, zorder=zorder_base+1)
    #ax.add_feature(cfeature.RIVERS, zorder=zorder_base+1)

    # Load and add South Korean administrative boundaries from shapefile
    shapefile_path = Path(figure_path, 'region', 'sig_20230729' ,'sig.shp')
    gdf = gpd.read_file(shapefile_path)

    # Set CRS if not already set
    if gdf.crs is None:
        gdf.set_crs(epsg=5179, inplace=True)

    # Transform CRS to WGS84 if necessary
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    gdf.boundary.plot(ax=ax, color='black', linewidth=boundary_lw, linestyle='-', zorder=zorder_base+5)

    # Set the extent to focus on South Korea
    ax.set_extent([x1, x2, y1, y2], crs=ccrs.PlateCarree())

def zoom_line(ax, axins, lw=1.5, edgecolor='black'):
    edgecolor = 'black'
    for spine in axins.spines.values():
        spine.set_edgecolor(edgecolor)
        spine.set_linewidth(lw)
    indicate = ax.indicate_inset_zoom(axins, zorder=110)
    indicate[0].set_linewidth(lw)
    indicate[0].set_edgecolor('black')
    indicate[0].set_alpha(1.0)
    indicate[0].set_linestyle('-')
    for line in indicate[1]:
        line.set_linewidth(1.0)
        line.set_color('black')
        line.set_alpha(1.0)