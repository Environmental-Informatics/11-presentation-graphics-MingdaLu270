#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 15:03:39 2020

@author: lu270
"""

## import useful tools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def ReadData( fileName ):
    """This function takes a filename as input, and returns a dataframe with
    raw data read from that file in a Pandas DataFrame.  The DataFrame index
    should be the year, month and day of the observation.  DataFrame headers
    should be "agency_cd", "site_no", "Date", "Discharge", "Quality". The 
    "Date" column should be used as the DataFrame index. The pandas read_csv
    function will automatically replace missing values with np.NaN, but needs
    help identifying other flags used by the USGS to indicate no data is 
    availabiel.  Function returns the completed DataFrame, and a dictionary 
    designed to contain all missing value counts that is initialized with
    days missing between the first and last date of the file."""
    
    # define column names
    colNames = ['agency_cd', 'site_no', 'Date', 'Discharge', 'Quality']

    # open and read the file
    DataDF = pd.read_csv(fileName, header=1, names=colNames,  
                         delimiter=r"\s+",parse_dates=[2], comment='#',
                         na_values=['Eqp'])
    DataDF = DataDF.set_index('Date')
    
    # replace negative value by nan
    DataDF['Discharge']=DataDF['Discharge'].mask(DataDF['Discharge']<0,np.nan)
    
    # quantify the number of missing values
    MissingValues = DataDF['Discharge'].isna().sum()
    
    
    return( DataDF, MissingValues)

def ClipData( DataDF, startDate, endDate ):
    """This function clips the given time series dataframe to a given range 
    of dates. Function returns the clipped dataframe and and the number of 
    missing values."""
    
    # locate the required data   
    #DataDF.index=pd.to_datetime(DataDF.index)
    #mask=(DataDF.index >= startDate) & (DataDF.index <= endDate)
    #DataDF=DataDF.loc[mask]
    
    DataDF=DataDF.loc[startDate:endDate]
    # Missing values for the specific period of data
    MissingValues=DataDF["Discharge"].isna().sum()
    
    return( DataDF, MissingValues )


def ReadMetrics( fileName ):
    """This function takes a filename as input, and returns a dataframe with
    the metrics from the assignment on descriptive statistics and 
    environmental metrics.  Works for both annual and monthly metrics. 
    Date column should be used as the index for the new dataframe.  Function 
    returns the completed DataFrame."""
    
    # read data by its filename
    DataDF=pd.read_csv(fileName,header=0,delimiter=',',parse_dates=['Date'],comment='#',index_col=['Date'])

    
    return( DataDF )


# the following condition checks whether we are running as a script, in which 
# case run the test code, otherwise functions are being imported so do not.
# put the main routines from your code after this conditional check.

if __name__ == '__main__':

    # define full river names as a dictionary so that abbreviations are not used in figures
    riverName = { "Wildcat": "Wildcat Creek",
                  "Tippe": "Tippecanoe River" }
    
    # define filename
    # define filenames as a dictionary
    # NOTE - you could include more than jsut the filename in a dictionary, 
    #  such as full name of the river or gaging site, units, etc. that would
    #  be used later in the program, like when plotting the data.
    fileName = { "Wildcat": "WildcatCreek_Discharge_03335000_19540601-20200315.txt",
                 "Tippe": "TippecanoeRiver_Discharge_03331500_19431001-20200315.txt" }
    
    Metrics = {"Annual": "Annual_Metrics.csv", "Monthly": "Monthly_Metrics.csv"}
    
    
    ## define a blank dictionary
    DataDF={}
    
    
    # clip data for the 5 water year period
    for file in fileName.keys():
        DataDF[file] = ReadData(fileName[file])
        DataDF[file] = ClipData(DataDF[file],startDate='2014-10-01',endDate='2019-09-30')
        
        # plot data
        plt.plot(DataDF[file]['Discharge'],label=riverName[file])
    plt.legend()
    plt.xlabel('Date',fontsize=20)
    plt.ylabel('Discharge (cfs)',fontsize=20)
    plt.title('Daily Streamflow Hydrograph',fontsize=20)
        
    # save plot
    plt.savefig('Daily Streamflow Hydrograph (2014-10-01 to 2019-09-30).png',dpi=96)
    plt.close()
        
    ## define a blank dictionary
    MetricsDF={}
    
    # read metrics data
    for file in Metrics.keys():
        MetricsDF[file]=ReadMetrics(Metrics[file])

    # plot coeff var
    
    plt.plot(MetricsDF['Annual'].loc[MetricsDF['Annual']['Station']=='Wildcat']['Coeff Var'],'bo')
    plt.plot(MetricsDF['Annual'].loc[MetricsDF['Annual']['Station']=='Tippe']['Coeff Var'],'ro')   
    plt.legend([riverName['Wildcat'],riverName['Tippe']])
    plt.xlabel('Date',fontsize=20)
    ax=plt.gca()
    ax.set_xticklabels(np.arange(1969,2019,10))

    plt.ylabel('Coefficient of Variation',fontsize=20)
    plt.title('Annual Coefficient of Variation',fontsize=20)
   
    plt.savefig('Annual Coefficient of Variation.png',dpi=96)
    
    plt.close()

        
    # plot Tqmean
    plt.plot(MetricsDF['Annual'].loc[MetricsDF['Annual']['Station']=='Wildcat']['TQmean'],'bo')
    plt.plot(MetricsDF['Annual'].loc[MetricsDF['Annual']['Station']=='Tippe']['TQmean'],'ro')   
    plt.legend([riverName['Wildcat'],riverName['Tippe']])
    plt.xlabel('Date',fontsize=20)
    ax=plt.gca()
    ax.set_xticklabels(np.arange(1969,2019,10))

    plt.ylabel('TQmean',fontsize=20)
    plt.title('Annual TQmean',fontsize=20)
   
    plt.savefig('Annual TQmean.png',dpi=96)       
    
    plt.close()

    # plot R-B index
    plt.plot(MetricsDF['Annual'].loc[MetricsDF['Annual']['Station']=='Wildcat']['R-B Index'],'bo')
    plt.plot(MetricsDF['Annual'].loc[MetricsDF['Annual']['Station']=='Tippe']['R-B Index'],'ro')   
    plt.legend([riverName['Wildcat'],riverName['Tippe']])
    plt.xlabel('Date',fontsize=20)
    ax=plt.gca()
    ax.set_xticklabels(np.arange(1969,2019,10))

    plt.ylabel('R-B Index',fontsize=20)
    plt.title('Annual R-B Index',fontsize=20)
   
    plt.savefig('Annual R-B Index.png',dpi=96)  
    plt.close()     
    
    # inport monthly data
    MonthlyData=ReadMetrics(Metrics['Monthly'])
    MonthlyData=MonthlyData.groupby('Station')
    
    for name, data in MonthlyData:
        columns=['Mean Flow']
        m=[3,4,5,6,7,8,9,10,11,0,1,2]
        index=0
        
        aveData=pd.DataFrame(0,index=range(1,13),columns=columns)
        
        # export data for plot
        for i in range(12):
            aveData.iloc[index,0]=data['Mean Flow'][m[index]::12].mean()
            index+=1
        
        # plot average annual monthly flow
        plt.scatter(aveData.index.values,aveData['Mean Flow'].values, label=riverName[name])
    plt.legend()
    plt.xlabel('Month',fontsize=20)
    plt.ylabel('Discharge (cfs)',fontsize=20)
    plt.title('Average Annual Monthly Flow',fontsize=20)
    plt.savefig('Average Annual Monthly Flow.png',dpi=96)
        
    plt.close()
        
        
    # Exceedance Probability calculation
    # import data
    epdata=ReadMetrics(Metrics['Annual'])
    # delete unneeded columns
    epdata=epdata.drop(columns=['site_no','Mean Flow','Median','Coeff Var','Skew','TQmean','R-B Index','7Q','3xMedian'])
    epdata=epdata.groupby('Station')
    
    # code from Cyber Training class
    for name, data in epdata:
        flow=data.sort_values('Peak Flow',ascending=False)
        ranks1=stats.rankdata(flow['Peak Flow'],method='average')
        ranks2=ranks1[::-1]
        
        ep=[100*(ranks2[i]/(len(flow)+1)) for i in range(len(flow))]
    
        # plot data
        plt.plot(ep,flow['Peak Flow'],label=riverName[name])
        # add grid lines to both axes
    plt.grid(which='both')
    plt.legend()
    plt.xlabel('Exceedance Probability (%)',fontsize=20)
    plt.ylabel('Dishcarge (cfs)',fontsize=20)
    plt.xticks(range(0,100,5))
    plt.tight_layout()
    plt.title('Exceedance Probability',fontsize=20)
    plt.savefig('Exceedance Probability.png',dpi=96)
    
    plt.close()    
        
        
        
        
        
        
        
        
        
        
        
        
        
        