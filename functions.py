import fiona
import rasterio
from rasterio.plot import show
import rasterio.mask
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def cv_downscaling(model, param_grid=None):
  if param_grid != None:
    grid = GridSearchCV(model, param_grid, cv=10)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    print(grid.best_params_)
  else:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

  print("MSE:\t", mean_squared_error(y_test, y_pred))
  print("R^2:\t", r2_score(y_test, y_pred))


def bbox(coord_list):
     box = []
     for i in (0,1):
         res = sorted(coord_list, key=lambda x:x[i])
         box.append((res[0][i],res[-1][i]))
     ret = f"({box[0][0]} {box[1][0]}, {box[0][1]} {box[1][1]})"
     return ret
# Add EE drawing method to folium.
def add_ee_layer(self, ee_object, vis_params, name):

    try:
        # display ee.Image()
        if isinstance(ee_object, ee.image.Image):
            map_id_dict = ee.Image(ee_object).getMapId(vis_params)
            folium.raster_layers.TileLayer(
            tiles = map_id_dict['tile_fetcher'].url_format,
            attr = 'Google Earth Engine',
            name = name,
            overlay = True,
            control = True
            ).add_to(self)

        # display ee.ImageCollection()
        elif isinstance(ee_object, ee.imagecollection.ImageCollection):
            ee_object_new = ee_object.mosaic()
            map_id_dict = ee.Image(ee_object_new).getMapId(vis_params)
            folium.raster_layers.TileLayer(
            tiles = map_id_dict['tile_fetcher'].url_format,
            attr = 'Google Earth Engine',
            name = name,
            overlay = True,
            control = True
            ).add_to(self)

        # display ee.Geometry()
        elif isinstance(ee_object, ee.geometry.Geometry):
            folium.GeoJson(
            data = ee_object.getInfo(),
            name = name,
            overlay = True,
            control = True
        ).add_to(self)

        # display ee.FeatureCollection()
        elif isinstance(ee_object, ee.featurecollection.FeatureCollection):
            ee_object_new = ee.Image().paint(ee_object, 0, 2)
            map_id_dict = ee.Image(ee_object_new).getMapId(vis_params)
            folium.raster_layers.TileLayer(
            tiles = map_id_dict['tile_fetcher'].url_format,
            attr = 'Google Earth Engine',
            name = name,
            overlay = True,
            control = True
        ).add_to(self)

    except:
        print("Could not display {}".format(name))

def clip_raster(path, out_name):
     with rasterio.open(path) as src:
         out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
         out_meta = src.meta
         print(out_meta)
     out_meta.update({"driver": "GTiff",
                      "height": out_image.shape[1],
                      "width": out_image.shape[2],
                      "transform": out_transform})
     # Output raster
     out_tif = os.path.join('/Users/hanni/PycharmProjects/Sattelite/', out_name)

     with rasterio.open(out_tif, "w", **out_meta) as dest:
         dest.write(out_image)
     return out_tif, out_image

def make_new_geotif(origional_path, clipped_array, new_name):
    n, xs, ys, gt, proj, dtype = gr.get_geo_info(origional_path)
    gtif = gr.create_geotiff(name =new_name, Array = clipped_array, driver = 'GTiff',
                                  ndv=n, xsize = xs, ysize = ys, geot =gt,
                                  projection = proj, datatype = dtype, band = 1)
    return gtif

def prepare_df(data1, colnames1,
               data2, colnames2,
               data3, colnames3,
               data4, colnames4,
               data5, colnames5,
               data6, colnames6):
  d = {colnames1: data1.flatten(),
       colnames2: data2.flatten(),
       colnames3: data3.flatten(),
       colnames4: data4.flatten(),
       colnames5: data5.flatten(),
       colnames6: data6.flatten(),
       }
  df = pd.DataFrame(data = d)
  # df = df.drop(['x', 'y'], axis=1)
  #df.columns = ['row', 'col', colnames, 'x', 'y']
  #df['coordinates'] = df['row'].astype('str') + df['col'].astype('str')
  # df = df.drop(['row', 'col'], axis=1)
  print(df)
  return df
