import fiona
import functions
import plot_maps as pm
from RandomForest import RandomForestModel
import DownscaleModel as DM

path_to_tif       = 'PATH/GPM-Downscale/Tif/'
path_to_shapefile = 'PATH/GPM-Downscale/iberia/iberia_merged.shp'
path_to_save_data = 'PATH/GPM-Downscale/downscaled_files/'

with fiona.open(path_to_shapefile, "r") as shapefile:
    shapes = [feature["geometry"] for feature in shapefile]

out_ndvi_11132m, img_ndvi_11132m   = functions.clip_raster(path_to_tif + 'ndvi_11132m_2020.tif', shapes, 'ndvi_11132m.tif')
out_gpm_11132m,  img_gpm_11132m    = functions.clip_raster(path_to_tif + 'gpm_11132m_2020.tif',  shapes, 'gpm_11132m.tif')
out_dem_11132m , img_dem_11132m    = functions.clip_raster(path_to_tif + 'dem_11132m_2020.tif',  shapes, 'nasadem_11132m.tif')
out_ndvi_1km    ,img_ndvi_1km    = functions.clip_raster(path_to_tif + 'ndvi_1km_2020.tif', shapes,  'ndvi_1km.tif')
out_dem_1km     ,img_dem_1km     = functions.clip_raster(path_to_tif + 'dem_1km_2020.tif'  , shapes, 'dem_1km.tif')

#fig, ax = pm.plot_input(out_gpm_11132m,out_ndvi_11132m,out_dem_11132m)
#fig.savefig('/Users/hanni/GPM/GPM-Downscale/figures/11132m_2020_annual_variables.png', dpi=300, bbox_inches='tight')


ndvi_low        = img_ndvi_11132m*0.0001
dem_low         = img_dem_11132m
gpm_low         = img_gpm_11132m*24*356
ndvi_high       = img_ndvi_1km
dem_high        = img_dem_1km

model_to_use    = RandomForestModel
downscale_array = DM.Downscale(ndvi_low  ,dem_low   ,gpm_low   ,ndvi_high ,dem_high  ,model_to_use,
                               dataset_path=path_to_save_data ).downscale_array

fig, ax = pm.plot_downscaled_array(downscale_array)
fig.savefig('PATH/GPM-Downscale/figures/1km_2020_annual_precipitation.png', dpi=300, bbox_inches='tight')
