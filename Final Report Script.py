# -*- coding: utf-8 -*-
"""
Created on %(1st July 2021 )s

@author: %(Mumtaz Ahmad)s
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#C:\Users\mumta\Desktop\Master\4th Sem\2.Microclimates\5.Final Exercise
#my Directory
mydir = 'C:/Users/mumta/Desktop/Master/4th Sem/2.Microclimates/5.Final Exercise/'

#reading forest csv file
dff = pd.read_csv(mydir+"Forest_20191205_20191217.csv", sep = ';',
                 header=0,
                 parse_dates=[0],
                 index_col=0)

#reading pasture csv file
dfp = pd.read_csv(mydir+"Pasture_20191205_20191217.csv", sep = ';',
                 header=0,
                 parse_dates=[0],
                 index_col=0)


#details of both files
Details_f = dff.describe()
Details_p = dfp.describe()


#combining forest and pasture file

dff = dff.rename(columns={'TA (degC)':'TA_F', 'RH (%)':'RH_F', 'SWC (%)':'SWC_F',
                          'TS (degC)':'TS_F', 'TKE (m2/s2)':'TKE_F', 'SW_IN (W/m2)': 'SWI_F', 
                          'SW_OUT (W/m2)':'SWO_F', 'LW_IN (W/m2)':'LWI_F', 'LW_OUT (W/m2)':'LWO_F',
                          'LE (W/m2)':'LE_F', 'H (W/m2)': 'H_F', 'G (W/m2)':'G_F',
                          'ALB (%)':'ALB_F', 'U (m/s)':'U_F', 'V (m/s)':'V_F'})
                          
                
dfp = dfp.rename(columns={'TA (degC)':'TA_P', 'RH (%)':'RH_P', 'SWC (%)':'SWC_P',
                          'TS (degC)':'TS_P', 'TKE (m2/s2)':'TKE_P', 'SW_IN (W/m2)': 'SWI_P', 
                          'SW_OUT (W/m2)':'SWO_P', 'LW_IN (W/m2)':'LWI_P', 'LW_OUT (W/m2)':'LWO_P',
                          'LE (W/m2)':'LE_P', 'H (W/m2)': 'H_P', 'G (W/m2)':'G_P',
                          'ALB (%)':'ALB_P', 'U (m/s)':'U_P', 'V (m/s)':'V_P'})
                          

df = pd.concat([dff,dfp], axis=1)



###########################################   1.  Mean diurnal cycle
#Air Temperature,Relative Humidity, Soil Temperature, Soil Water Content

df3H = df.resample('3H').mean()

df3H['Time'] = df3H.index.map(lambda x: x.strftime("%H:00"))
df3H_gb = df3H.groupby('Time').describe()

#plotting

fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(421)
ax2 = fig.add_subplot(422)
ax3 = fig.add_subplot(423)
ax4 = fig.add_subplot(424)

ax1.plot(df3H_gb.index,df3H_gb['TA_F']['mean'], 'g', label='Forest', linestyle = 'solid', linewidth = 2)
ax1.plot(df3H_gb.index,df3H_gb['TA_P']['mean'], 'r', label='Pasture', linestyle = 'solid', linewidth = 2)

ax2.plot(df3H_gb.index,df3H_gb['RH_F']['mean'], 'r', label='Forest', linestyle = 'solid', linewidth = 2)
ax2.plot(df3H_gb.index,df3H_gb['RH_P']['mean'], 'b', label='Pasture', linestyle = 'solid', linewidth = 2)

ax3.plot(df3H_gb.index,df3H_gb['TS_F']['mean'], 'b', label='Forest', linestyle = 'solid', linewidth = 2)
ax3.plot(df3H_gb.index,df3H_gb['TS_P']['mean'], 'g', label='Pasture', linestyle = 'solid', linewidth = 2)

ax4.plot(df3H_gb.index,df3H_gb['SWC_F']['mean'], 'b', label='Forest', linestyle = 'solid', linewidth = 2)
ax4.plot(df3H_gb.index,df3H_gb['SWC_P']['mean'], 'g', label='Pasture', linestyle = 'solid', linewidth = 2)

ax1.legend()
ax1.set_title(' Air Temperature ', fontsize=15)
ax1.set_xlabel('Time')
ax1.set_ylabel('TA (degC)')

ax2.legend()
ax2.set_title('Relative Humidity', fontsize=15)
ax2.set_xlabel('Time')
ax2.set_ylabel('RH (%)')

ax3.legend()
ax3.set_title(' Soil Temperature', fontsize=15)
ax3.set_xlabel('Time')
ax3.set_ylabel('TS (degC)')

ax4.legend()
ax4.set_title(' Soil Water Content', fontsize=15)
ax4.set_xlabel('Time')
ax4.set_ylabel('SWC (%)')

plt.tight_layout()



#########################################  2.   Mean wind field

#daily mean
dfD = df.resample('D').mean()

dfD['Day'] = dfD.index.map(lambda x: x.strftime('%d'))
dfD_gb = dfD.groupby('Day').describe()


fig = plt.figure(figsize=(14,6))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)


ax1.plot(dfD_gb.index,dfD_gb['U_F']['mean'], 'b', label='Forest', linestyle = 'solid', linewidth = 2)
ax1.plot(dfD_gb.index,dfD_gb['U_P']['mean'], 'r', label='Pasture', linestyle = 'solid', linewidth = 2)

ax2.plot(dfD_gb.index,dfD_gb['V_F']['mean'], 'b', label='Forest', linestyle = 'solid', linewidth = 2)
ax2.plot(dfD_gb.index,dfD_gb['V_P']['mean'], 'r', label='Pasture', linestyle = 'solid', linewidth = 2)


ax1.legend()
ax1.set_title('Zonal wind field', fontsize=15)
ax1.set_xlabel('Day')
ax1.set_ylabel('U (m/s)')

ax2.legend()
ax2.set_title('Meridional wind field', fontsize=15)
ax2.set_xlabel('Day')
ax2.set_ylabel('V (m/s)')

plt.tight_layout()



#######################################     3.  Albedo and Bowen Ratio

#Albedo
af = df['ALB_F']/100
ap = dfp['ALB_P']/100

#Bowen Ratio
brf = df['H_F']/dff['LE_F']
brp = df['H_P']/dfp['LE_P']


#plotting
fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)


ax1.plot(af, 'g', label='Forest')
ax1.plot(ap, 'r', label='Pasture')

ax2.plot(brf, 'g', label='Forest')
ax2.plot(brp, 'b', label='Pasture')

ax1.legend()
ax1.set_title(' Albedo ' )
ax1.set_xlabel('Date')
ax1.set_ylabel('value')

ax2.legend()
ax2.set_title('Bowen Ratio')
ax2.set_xlabel('Date')
ax2.set_ylabel('value')

plt.tight_layout()



####################################   4. T / RH and TS / SWC.

#plotting

#Air Temperature and Relative Humidity, 

fig, (tarh, tssw) = plt.subplots(2,1, figsize=(13,6), sharex=True)
fig.subplots_adjust(hspace=0.1)

tarh.plot(df['TA_F'], label ='Forest',color = 'black', linestyle = 'solid', linewidth = 2)
tarh.plot(df['TA_P'], label ='Pasture',color = 'black', linestyle = '--', linewidth = 2)
tarh.set_ylabel('Air Temperature (ᵒC)', color ='black', fontsize=15)
tarh.set_yticks([10,15,20,25])
tarh.tick_params(axis='y', labelcolor='black', labelsize=15)
tarh.patch.set_visible(False)
tarh2 = tarh.twinx()
tarh2.plot(df['RH_F'], label ='Forest',color='blue', linestyle = 'solid', linewidth = 2)
tarh2.plot(df ['RH_P'], label ='Pasture',color='blue', linestyle = '--', linewidth = 2)
tarh2.set_ylabel('Relative Humidity (%)', color ='blue', fontsize=15)
tarh2.set_yticks([20,40,60,80,100])
tarh2.tick_params(axis='y', labelcolor='blue', labelsize=15)
tarh2.set_zorder(tarh.get_zorder()-1)

# Soil temperature ,Soil water content

tssw.plot(df['TS_F'], label ='Forest',color='red',linestyle = 'solid',linewidth=2)
tssw.plot(df['TS_P'], label ='Pasture',color='red', linestyle = '--',linewidth=2)
tssw.set_ylabel('Surface Temperature (ᵒC)', color ='red', fontsize=15)
tssw.set_yticks([15,16,17,18,19,20])
tssw.tick_params(axis='y', labelcolor='red', labelsize=15)
tssw.set_xlabel('Date', fontsize=15)
tssw.patch.set_visible(False)
tssw2 = tssw.twinx()
tssw2.plot(df['SWC_F'], label ='Forest',color='green', linestyle = 'solid', linewidth = 2)
tssw2.plot(df['SWC_P'], label ='Pasture',color='green', linestyle = '--', linewidth = 2)
tssw2.set_ylabel('Soil Water Content (%)', color ='green', fontsize=15)
tssw2.set_yticks([0,10,20,30,40,50])
tssw2.tick_params(axis='y', labelcolor='green', labelsize=15)
tssw.tick_params(axis='x', labelsize=15)
tssw2.set_zorder(tarh.get_zorder()-1)

tarh.legend(frameon = False, loc="lower center", ncol = 2, fontsize='large')
tarh2.legend(frameon = False, loc="lower left", ncol = 2, fontsize='large')
tssw.legend(frameon = False, loc="lower center", ncol = 2, fontsize='large')
tssw2.legend(frameon = False, loc="lower left", ncol = 2, fontsize='large')




####################################   4. net-radiation 

#plotting

fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(421)
ax2 = fig.add_subplot(422)
ax3 = fig.add_subplot(423)
ax4 = fig.add_subplot(424)



ax1.plot(dfD_gb.index,dfD_gb['SWI_F']['mean'], 'g', label='Forest', linestyle='--', linewidth = 2)
ax1.plot(dfD_gb.index,dfD_gb['SWI_P']['mean'], 'r', label='Pasture', linewidth = 2)

ax2.plot(dfD_gb.index,dfD_gb['SWO_F']['mean'], 'r', label='Forest', linestyle='--')
ax2.plot(dfD_gb.index,dfD_gb['SWO_P']['mean'], 'b', label='Pasture', linewidth = 2)

ax3.plot(dfD_gb.index,dfD_gb['LWI_F']['mean'], 'b', label='Forest', linestyle='--')
ax3.plot(dfD_gb.index,dfD_gb['LWI_P']['mean'], 'g', label='Pasture', linewidth = 2)

ax4.plot(dfD_gb.index,dfD_gb['LWO_F']['mean'], 'b', label='Forest', linestyle='--')
ax4.plot(dfD_gb.index,dfD_gb['LWO_P']['mean'], 'g', label='Pasture', linewidth = 2)

ax1.legend()
ax1.set_title('Shortwave incoming' )
ax1.set_xlabel('Day')
ax1.set_ylabel('SW_IN (W/m2)')

ax2.legend()
ax2.set_title('Shortwave outgoing')
ax2.set_xlabel('Day')
ax2.set_ylabel('SW_OUT (W/m2)')

ax3.legend()
ax3.set_title('Longwave incoming')
ax3.set_xlabel('Day')
ax3.set_ylabel('LW_IN (W/m2)')

ax4.legend()
ax4.set_title('Longwave outgoing')
ax4.set_xlabel('Day')
ax4.set_ylabel('LW_OUT (W/m2)')


plt.tight_layout()



#net-radiation plotting


#daily(For better view Daily plotting is preferable)
dfD= df.resample('D').mean()

RnDf = (dfD["SWI_F"] + dfD["SWO_F"]) - (dfD["LWO_F"] - dfD["LWI_F"])

RnDp = (dfD["SWI_P"] + dfD["SWO_P"]) - (dfD["LWO_P"] - dfD["LWI_P"])


fig = plt.figure(figsize=(10,2))
ax1 = fig.add_subplot(111)
ax1 = fig.add_subplot(111)


ax1.plot(RnDf, 'g', label='Rn forest')
ax1.plot(RnDp, 'r', label='Rn pasture')


ax1.legend()
ax1.set_title('Net Radiation of Forest and Pasture ' )
ax1.set_xlabel('Date')
ax1.set_ylabel('Value')

plt.tight_layout()






#############################################  5. surface fluxes



Rnf = (df["SWI_F"] + df["SWO_F"]) - (df["LWO_F"] - df["LWI_F"])
Rnp = (df["SWI_P"] + df["SWO_P"]) - (df["LWO_P"] - df["LWI_P"])



fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)


ax1.plot(Rnf, 'g', label='Net Radiation',linewidth = 2)
ax1.plot(df['LE_F'], 'r', label='Latent heat flux',linewidth = 2)
ax1.plot(df['SWC_F'], 'b', label='Soil water content',linewidth = 2)


ax2.plot(Rnp, 'g', label='Net Radiation',linewidth = 2)
ax2.plot(df['LE_F'], 'r', label='Latent heat flux',linewidth = 2)
ax2.plot(df['SWC_F'], 'b', label='Soil water content',linewidth = 2)


ax1.legend()
ax1.set_title(' Forest ' )
ax1.set_xlabel('Date')
ax1.set_ylabel('Value')

ax2.legend()
ax2.set_title('Pasture')
ax2.set_xlabel('Date')
ax2.set_ylabel('Value')


plt.tight_layout()



#############################################  6. TKE



fig = plt.figure(figsize=(8,2))
ax1 = fig.add_subplot(111)
ax1 = fig.add_subplot(111)


ax1.plot(dfD['TKE_F'], 'g', label='Forest',linewidth = 2)
ax1.plot(dfD['TKE_P'], 'r', label='Pasture',linewidth = 2)


ax1.legend()
ax1.set_title('Turbulent kinetic energy ' )
ax1.set_xlabel('Date')
ax1.set_ylabel('Value')

plt.tight_layout()



##############################################    7. large-scale condition

#Large-scale conditions
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
from matplotlib.cm import get_cmap
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib import dates as d
import datetime as dt
import matplotlib.image as image
from windrose import WindroseAxes

#reading ER5_2019Dec05-17_day.nc file
mydir = 'C:/Users/mumta/Desktop/Master/4th Sem/2.Microclimates/5.Final Exercise/'
file = 'ER5_2019Dec05-17_day.nc' # netcdf file
dfnc = xr.open_dataset(mydir+file) # open netcdf file using xarray
dfnc.info() 



#Given Station Coordinates
pas_lon, pas_lat = -79.0755, -3.96670
fors_lon, fors_lat = -79.07515638888889, -3.9737247222222223

r = dfnc.r
u = dfnc.u
v = dfnc.v

lon = dfnc.longitude
lat = dfnc.latitude

humid_levels = np.arange(0., 100., 6.)
dxTXT = {0: 'Night', 1:'Day'}

u = np.array(dfnc.variables['u'][:],dtype=np.float32)
v = np.array(dfnc.variables['v'][:],dtype=np.float32)
X, Y  = np.meshgrid(lon,lat)
uu = X + Y**2
speed = np.sqrt(u**2 + v**2)

def plotRH(dn,time):
    dn.add_feature(cf.COASTLINE.with_scale('50m'), linewidth=0.5, zorder=4, edgecolor='k')
    dn.add_feature(cf.BORDERS.with_scale('50m'), linewidth=0.5, zorder=4)
    dn.stock_img()
    dn_rh_contours = dn.contourf(lon,lat,r[time,:,:],levels=humid_levels,cmap=get_cmap("rainbow"),
                             transform=ccrs.PlateCarree())
    cb_dn = plt.colorbar(dn_rh_contours, ax = dn, orientation = 'vertical',
                  fraction=0.040,pad=-0.01,shrink=0.70,ticks=[0,20,40,60,80,100,120])
    cb_dn.ax.tick_params(labelsize=11)
    cb_dn.ax.set_title('%',pad=4.0,ha='center',fontsize=11)

    
    dn.set_xticks(np.arange(-82.5, -60, 4),crs=ccrs.PlateCarree())
    dn.set_yticks(np.arange(-10, 10, 4),crs=ccrs.PlateCarree())
    dn.xaxis.set_major_formatter(LongitudeFormatter())
    dn.yaxis.set_major_formatter(LatitudeFormatter())
    dn.tick_params(reset=True,axis='both',which='major',labelsize=10,direction='in',bottom = True,
                top = True,left = True, right = True,width = 0.5, labelbottom=True, zorder=12) 
    dn.outline_patch.set_linewidth(0.2)
    dn.outline_patch.set_zorder(6)
    dn.set_xlim(lon[0],lon[-1])
    dn.set_ylim(lat[-1],lat[0])
    dn.text(0.03,1.05, dxTXT[time],transform=dn.transAxes,fontsize=10,
                color = 'black', ha='left',
           bbox={'facecolor': 'white'})

    d_q = dn.quiver(lon[::8],lat[::8],dfnc.u[time,::8, ::8], dfnc.v[time,::8,::8],
                 color='red')
    
    dn.quiverkey(d_q, X=0.3, Y=1.06, U=10,
             label='Quiver key, length = 10',labelpos='E',
             color='red')
    dn_wd_speed = dn.streamplot(X,Y,u[time],v[time], color=speed[time], linewidth=2, cmap = get_cmap('gnuplot2_r'),
                  transform=ccrs.PlateCarree())
    
    dn_wd = fig.colorbar(dn_wd_speed.lines,orientation='horizontal',
                         fraction=0.030,pad=0.15,shrink=0.8,
                         ticks=[0,1,2,3,4,5,6,7,8])
    
    dn_wd.ax.tick_params(labelsize=11)
    dn_wd.ax.set_title('m/s',pad=3.0,ha='center',fontsize=11)
    
    dn.text(pas_lon-1, pas_lat+2.2,'Ecuador', ha='left', fontsize = 12)
    dn.text(pas_lon+8, pas_lat+12.6,'Relative Humidity', ha='right', fontsize = 12, color='black')
    dn.plot([pas_lon, fors_lon], [pas_lat, fors_lat], color='red', marker='o', markersize = '5',
        transform=ccrs.PlateCarree())
lat.values

#figure
fig = plt.figure(figsize=(8,6))
fig.subplots_adjust(hspace=0.14)
fig.tight_layout()
gs = fig.add_gridspec(2,2,height_ratios=[1,1])

for i in range(0,2):
    sfig = fig.add_subplot(gs[i],projection=ccrs.PlateCarree())
    plotRH(sfig,i)




##############################################    8. windrose

def wind_speed_dir(u,v):
    wspd = np.sqrt(u*u + v*v)
    wdir = 270. - (np.arctan2(u,v) * 180. / np.pi)
    return wspd, wdir

dfncuv= dfnc.sel(time=slice("2019-12-05", "2019-12-17"))



dfncuv = dfncuv.to_dataframe()


dfncuv['wspd'], dfncuv['wdir'] = wind_speed_dir(dfncuv['u'], dfncuv['v'])



fig = plt.figure(figsize=(10,10))
fig.subplots_adjust(hspace=0.3,wspace=0.3)

ax1 = fig.add_subplot(1,1,1,projection='windrose')

ax1.bar(dfncuv.wdir, dfncuv.wspd, bins=np.arange(0,8.0,1),cmap = get_cmap('gist_ncar'),edgecolor='black')
ax1.legend(bbox_to_anchor=(-0.15, 1.08),fontsize = 18, ncol=4)
ax1.tick_params(labelsize=20)






