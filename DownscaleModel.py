import xarray as xr
import functions
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


class Downscale:
    def __init__(self, ndvi_low, dem_low, gpm_low,ndvi_high,dem_high, model_to_use, dataset_path):
        self.ndvi_low = ndvi_low
        self.dem_low  = dem_low
        self.gpm_low  = gpm_low
        self.ndvi_high  = ndvi_high
        self.dem_high   = dem_high
        self.model_to_use = model_to_use
        self.dataset_path = dataset_path

        data_for_modeling = functions.prepare_df(ndvi_low, 'ndvi',
                                                 dem_low, 'dem',
                                                 gpm_low, 'precipitation',
                                                 )
        #fig, ax =pm.plot_correlations(data_for_modeling['precipitation'], data_for_modeling['ndvi'], data_for_modeling['dem'])
        #fig.savefig('/Users/hanni/GPM/GPM-Downscale/figures/11132m_2021_annual_variables_correlations.png', dpi=300, bbox_inches='tight')
        X = data_for_modeling[['ndvi', 'dem']].values[:,:]
        y = data_for_modeling[['precipitation']].values[:,:]

        model              = model_to_use(X, y).best_model


        data_for_prediction = functions.get_data_for_prediction( ndvi_high , dem_high)

        X                   = data_for_prediction[['ndvi', 'dem']]
        X_scaled            = StandardScaler().fit_transform(X)
        y_pred              = model.predict(X_scaled)

        data_for_prediction['precipitation'] = y_pred
        final_1km_df                         = data_for_prediction[['row', 'col', 'ndvi', 'dem','precipitation']]
        downscale_result                     = final_1km_df[['row', 'col', 'precipitation']]

        self.downscale_array = np.zeros(shape=(960, 1433), dtype=float)
        row_col         = downscale_result[['row', 'col', 'precipitation']]

        for i in tqdm(row_col.iterrows()):
          self.downscale_array[int(i[1]['row']), int(i[1]['col'])] = i[1]['precipitation']

        self.downscale_array[self.downscale_array == 0.0] = np.nan

        lat = np.linspace(35.176807, 43.791248, num=960, endpoint=True, retstep=False, dtype=None, axis=0)#.round(1)
        lon = np.linspace(-9.546804, 3.322638 , num=1433, endpoint=True, retstep=False, dtype=None, axis=0)#.round(1)

        ds = xr.Dataset({
        'precipitation': xr.DataArray(
               data   = self.downscale_array.T,   # enter data here
               dims   = ['lon', 'lat'],
               coords = {'lon': lon, 'lat': lat},
               attrs  = {
                   'long_name': 'Mean annual precipitaiton (mm/y)',
                   'units'     : ''
                   }
               ),
        },

        attrs={'description': 'Global Precipitation Measurement (GPM) v6 from ee.ImageCollection("NASA/GPM_L3/IMERG_MONTHLY_V06")',
               'location'   : 'Iberian peninsula',
               'reference'  : 'Johanna Roschke, roschke.johanna@web.de'}
        )


        ds.to_netcdf(self.dataset_path + f'2020_barbados_clouds.nc')

