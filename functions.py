import georasters as gr
import pandas as pd
import rasterio
import rasterio.mask
import os
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def cv_downscaling(model, X_train, y_train , X_test, y_test, param_grid=None):
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
  return model, y_pred
def prepare_df(data1, colnames1,
               data2, colnames2,
               data3, colnames3,
               ):
  d = {colnames1: data1.flatten(),
       colnames2: data2.flatten(),
       colnames3: data3.flatten(),

       }
  df = pd.DataFrame(data = d)


  return df
def prepare_data(X, y):
    # Prepare the X and y


    # Standardize the data
    scaler = StandardScaler()
    X_scaled_original = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled_original, y, test_size=0.1, random_state=42)

    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    return X_train, X_test, y_train,y_test , X_scaled_original
def bbox(coord_list):
     box = []
     for i in (0,1):
         res = sorted(coord_list, key=lambda x:x[i])
         box.append((res[0][i],res[-1][i]))
     ret = f"({box[0][0]} {box[1][0]}, {box[0][1]} {box[1][1]})"
     return ret
def flatten_df(data, colname):
    df = pd.DataFrame(data[0,:,:])
    df = df.stack().reset_index()
    df.columns = ['row','col', colname]
    return df
def get_data_for_prediction(ndvi_1km, dem_1km):

    df_ndvi_1km = flatten_df(ndvi_1km, 'ndvi')
    df_dem_1km  = flatten_df(dem_1km, 'dem')
    df_dem_1km  = df_dem_1km.drop(['row', 'col'], axis = 1)

    data_for_prediction = pd.concat([df_ndvi_1km, df_dem_1km ], axis=1)
    return data_for_prediction
def clip_raster(path,shapes, out_name):
     with rasterio.open(path) as src:
         out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
         out_meta = src.meta

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
