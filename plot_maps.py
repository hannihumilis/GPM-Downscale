import rasterio
import matplotlib.pyplot as plt
import numpy as np
import utils
import cmaps
import pandas as pd

def plot_input(out_gpm_11132m, out_ndvi_11132m, out_dem_11132m):
    # Visualize
    b0 = rasterio.open(out_gpm_11132m)
    b1 = rasterio.open(out_ndvi_11132m)
    b2 = rasterio.open(out_dem_11132m)

    ba0 = b0.read()
    ba1 = b1.read()
    ba2 = b2.read()
    ba0[ba0 == 0] = np.nan
    ba1[ba1 == 0] = np.nan
    ba2[ba2 == 0] = np.nan

    bres0 =np.reshape(ba0*24*365, (ba0.shape[0]*ba0.shape[1], ba0.shape[2]))
    bres1 =np.reshape(ba1*0.0001, (ba1.shape[0]*ba1.shape[1], ba1.shape[2]))
    #bres2 =np.reshape(ba2, (ba2.shape[0]*ba2.shape[1], ba2.shape[2]))
    ba2 = ba2.astype('float')
    ba2[ba2 == 0] = np.nan
    bres2 = ba2[0,:,:]


    lat_ticks = np.linspace(35.176807, 43.791248,  num=11, endpoint=True, retstep=False, dtype=None, axis=0)#.round(1)
    lon_ticks = np.linspace(-9.546804, 3.322638 ,  num=9, endpoint=True, retstep=False, dtype=None, axis=0)#.round(1)
    formatted_lat, formatted_lon = utils.format_lat_lon(lat_ticks , lon_ticks )


    fig = plt.figure(figsize=(15, 20))

    ax1 = fig.add_subplot(3,1,1)
    ax2 = fig.add_subplot(3,1,2)
    ax3 = fig.add_subplot(3,1,3)

    base0 = ax1.imshow( bres0, cmap= cmaps.precip, )
    base1 = ax2.imshow( bres1, cmap= cmaps.ndvi_map)
    base2 = ax3.imshow( bres2, cmap= cmaps.terras)

    cbar1 = plt.colorbar(base0, ax=ax1, fraction=0.03, pad=0.05)
    cbar2 = plt.colorbar(base1, ax=ax2, fraction=0.03, pad=0.05)
    cbar3 = plt.colorbar(base2, ax=ax3, fraction=0.03, pad=0.05)
    cbar1.ax.tick_params(labelsize=16)
    cbar2.ax.tick_params(labelsize=16)
    cbar3.ax.tick_params(labelsize=16)

    cbar1.set_label(label='Precipitation (mm/year)' , size=16 )
    cbar2.set_label(label='NDVI'                    , size=16 )
    cbar3.set_label(label='Elevation (m)'           , size=16 )

    for ax in (ax3,ax1,ax2):
        ax.set_ylabel('Longitude (°)', fontsize = 16)
        ax.set_xlabel('Latitude (°)' , fontsize = 16)
        ax.set_yticklabels(formatted_lat)
        ax.set_xticklabels(formatted_lon)
        ax.grid(':', alpha = 0.4)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(16)


    ax1.set_title('GPM: Monthly Global Precipitation Measurement v6 (11132m) ' , fontsize = 16)
    ax3.set_title('SRTM Digital Elevation Data Version 4 (11132m)'             , fontsize = 16)
    ax2.set_title('MOD13Q1.061 Terra Vegetation Indices 16-Day Global (11132m)', fontsize = 16)

    plt.tight_layout()
    return fig, ax

def plot_correlations(precip, ndvi, dem):
    fig,( ax1, ax2 )= plt.subplots(ncols = 2, figsize = (15,5))
    ax1.scatter( ndvi, precip, alpha = 0.4)
    ax2.scatter(  dem, precip, alpha = 0.4)
    ax1.set_xlabel('NDVI' , fontsize = 16)
    ax2.set_xlabel('Elevation (m)' , fontsize = 16)

    for ax in (ax1,ax2):
            ax.set_ylabel('Precipitation (mm/y)', fontsize = 16)

            ax.grid(':', alpha = 0.4)
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(16)
    plt.tight_layout()
    return fig, ax
def plot_feature_importance(best_model):
    std = np.std([tree.feature_importances_ for tree in best_model.estimators_], axis=0)
    forest_importances = pd.Series(best_model.feature_importances_, index=['ndvi', 'dem'])
    ft = 18
    fig, ax = plt.subplots(1, 1, figsize = (5,5))
    forest_importances.plot.bar(yerr=std, ax=ax)
    #ax.set_title("Feature importances", fontsize = ft,  weight ='bold')
    ax.set_ylabel('Feature importances', fontsize = ft, weight ='bold')
    ax.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    ax.set_yticklabels([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], fontsize = 14)
    ax.set_xticks([0,1])
    ax.set_xticklabels(['NDVI', 'DEM'], fontsize = 14)
    plt.tight_layout()
    return fig, ax
def plot_downscaled_array(downscale_array):
    fig = plt.figure(figsize=(30, 10))
    ax1 = fig.add_subplot(122)
    #base0 = ax.matshow(downscale_array*24*365, cmap=precip)
    # plot annual precipitation values
    base0 = ax1.imshow(downscale_array*24*365, cmap = cmaps.precip )
    return fig, ax1
