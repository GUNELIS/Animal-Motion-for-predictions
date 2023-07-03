import pandas as pd
import folium.plugins as plugins
import rasterio
from rasterio.transform import Affine



def preprocess_sperm_whale(path):
    sperm_whales = pd.read_csv(path)
    temp_df = sperm_whales['event-id'].str.split(',', expand=True)
    temp_df.columns = ['event-id', 'visible', 'timestamp', 'location-long', 'location-lat',
       'gps:hdop', 'gps:satellite-count', 'manually-marked-outlier',
       'study-specific-measurement', 'sensor-type',
       'individual-taxon-canonical-name', 'tag-local-identifier',
       'individual-local-identifier', 'study-name']
    temp_df = temp_df.rename(columns={'location-long': 'longitude', 'location-lat': 'latitude', 'tag-local-identifier': 'name'})
    sperm_whales =temp_df.iloc[:,[0,1,2,3,4,12]].copy() # refine the csv and get only the desired columns from og dataset
    # sperm_whales.to_csv('Sperm_whales_refined.csv', index=False)

    for i, row in sperm_whales.iterrows():
        sperm_whales.at[i, "longitude"] = float(row["longitude"])
        sperm_whales.at[i, "latitude"] = float(row["latitude"])

    sperm_whales = sperm_whales.rename(columns={'timestamp': 'date'})
    sperm_whales = sperm_whales.rename(columns={'individual-local-identifier': 'name'})
    sperm_whales['date'] = pd.to_datetime(sperm_whales['date']) 
    sperm_whales['year'] = sperm_whales['date'].dt.year
    sperm_whales['depth'] = sperm_whales.apply(lambda row: get_sea_depth(row['latitude'], row['longitude']),axis=1)
    sperm_whales['species'] = 'Physeter Macrocephalus'
    sperm_whales = sperm_whales.drop('event-id',axis=1)
    sperm_whales = sperm_whales.drop('visible',axis=1)
    return sperm_whales

def preprocess_all_animals(data):
    data['date'] = pd.to_datetime(data['date'])
    return data

def preprocess_blue_whale(path):
    blue_whales = pd.read_csv(path)
    blue_whales =blue_whales.iloc[:,[0,1,2,3,4,18]].copy() # refine the csv and get only the desired columns from og dataset
    blue_whales = blue_whales.rename(columns={'location-long': 'longitude', 'location-lat': 'latitude', 'tag-local-identifier': 'name'})

    for i, row in blue_whales.iterrows():
        blue_whales.at[i, "longitude"] = float(row["longitude"])
        blue_whales.at[i, "latitude"] = float(row["latitude"])


    blue_whales = blue_whales.rename(columns={'timestamp': 'date'})
    blue_whales = blue_whales.rename(columns={'individual-local-identifier': 'name'})
    blue_whales['date'] = pd.to_datetime(blue_whales['date'])
    blue_whales['year'] = blue_whales['date'].dt.year

    blue_whales['depth'] = blue_whales.apply(lambda row: get_sea_depth(row['latitude'], row['longitude']),axis=1)
    blue_whales['species'] = 'Balaenoptera Musculus'
    blue_whales = blue_whales.drop('event-id',axis=1)
    blue_whales = blue_whales.drop('visible',axis=1)


    return blue_whales



"""
This function proccesses Tiger shark data from the DataONE website.
It also adds the new columns:
year - the year of observation
depth - the water depth in meters where the animal was swimming

Data Acknowledgements: 

Movement Patterns and Habitat Use of Tiger Sharks Across Ontogeny in the Gulf of Mexico, 2010-2018
Published in 2020 and 
written by Matthew Ajemian, Neil Hammerschlag, Marcus Drymon, David Wells, and Brett Falterman
https://search.dataone.org/view/10.24431/rw1k44e

"""
def preprocess_tiger_sharks(path):    
    tiger_sharks = pd.read_csv(path)
    tiger_sharks['datetime '] = pd.to_datetime(tiger_sharks['datetime '])
    tiger_sharks = tiger_sharks.rename(columns = {'datetime ':'date', 
                                                  'lon':'longitude', 
                                                  'lat': 'latitude',
                                                  'id':'name'})
    tiger_sharks['year'] = tiger_sharks['date'].dt.year
    tiger_sharks['depth'] = tiger_sharks.apply(lambda row: get_sea_depth(row['latitude'], row['longitude']),axis=1)
    tiger_sharks['species'] = 'Galeocerdo Cuvier'
    tiger_sharks = tiger_sharks.drop('lc',axis=1)
    tiger_sharks = tiger_sharks[['date', 'longitude', 'latitude', 'name', 'year', 'depth', 'species']]

    return tiger_sharks


def preprocess_WA_sharks(path):    
    WA_sharks = pd.read_csv(path)
    WA_sharks['Date'] = pd.to_datetime(WA_sharks['Date'])
    WA_sharks = WA_sharks.rename(columns = {'Date':'date', 
                                            'Latitude_DD':'longitude', 
                                            'Longitude_DD': 'latitude',
                                            'RotoTag':'name',
                                            'Species':'species'})

    WA_sharks['year'] = WA_sharks['date'].dt.year
    WA_sharks['depth'] = WA_sharks.apply(lambda row: get_sea_depth(row['latitude'], row['longitude']),axis=1)
    WA_sharks = WA_sharks.drop(['Sex','Cal','TL','TL_Notes','FL','PCL','SpaghettiTag','Notes'],axis=1)
    WA_sharks = WA_sharks[['date', 'longitude', 'latitude', 'name', 'year', 'depth', 'species']]

    return WA_sharks

def preprocess_whale_sharks(path):
    whale_sharks = pd.read_csv(path)
    for i, row in whale_sharks.iterrows():
            whale_sharks.at[i, "longitude"] = float(row["longitude"])
            whale_sharks.at[i, "latitude"] = float(row["latitude"])

    whale_sharks = whale_sharks.rename(columns={'timestamp': 'date'})
    whale_sharks = whale_sharks.rename(columns={'id_tag': 'name'})
    whale_sharks['date'] = pd.to_datetime(whale_sharks['date'])
    whale_sharks = whale_sharks.drop('month' , axis=1)
    whale_sharks['year'] = whale_sharks['date'].dt.year

    whale_sharks['depth'] = whale_sharks.apply(lambda row: get_sea_depth(row['latitude'], row['longitude']),axis=1)
    whale_sharks['species'] = 'Whale Sharks Rhincodon Typus'
    whale_sharks = whale_sharks.drop('event-id',axis=1)
    whale_sharks = whale_sharks.drop('visible',axis=1)


    return whale_sharks



###################################### Weather Data ################################################3    
        

def preProcess_weather_data(data):
    """
    The weather data contains coordinates in the format of N and W. eg: 32.43N and 23.3W
    This function makes all N & E positive and S & W negative 
    """
    
    if isinstance(data['date'].iloc[0], str):
        data['date'] = data['date'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
        
    lat_column = data['latitude']
    lon_column = data['longitude']

    for i in range(len(lat_column)):
        value = lat_column.iloc[i]
        if "N" in value:
            lat = float(value.strip("N"))
        elif "S" in value:
            lat = -1 * float(value.strip("S"))
        lat_column.iloc[i] = lat
    for i in range(len(lon_column)):
        value = lon_column.iloc[i]
        if "E" in value:
            lon = float(value.strip("E"))
        elif "W" in value:
            lon = -1 * float(value.strip("W"))
        lon_column.iloc[i] = lon
    data['latitude'] = lat_column
    data['longitude'] = lon_column    
    data['year'] = data['date'].dt.year

    return data

def process_hurricane_txt(file_path):
    """
    This function reads the hurricane text file taken from the National Hurricane Center nhc.noaa.gov
    The only parameter is the text file path itself and the functions
    returns a pandas data frame with relevant columns in correct form.

    This link for data: https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2020-052520.txt
    """
    hurricane_columns = ['date', 'Hour UTC', 'RI', 'HU', 'latitude', 'longitude', 'max_wind', 'min_pressure', 'ne34', 'se34', 'sw34', 'nw34', 'ne50', 'se50', 'sw50', 'nw50', 'ne64', 'se64', 'sw64', 'nw64', 'name']

    hurricane_info = []
    current_hurricane = None
    names_dict = {}  # Keep track of hurricane names and suffixes
    with open(file_path, 'r') as f:
        for line in f:
            # if this is a new hurricane
            if line.startswith('AL'): # AL is a starter indicating new hurricane information
                name = line.split(',')[1].strip() # get the name which is after the comma
                suffix = names_dict.get(name, 1)  # Get the suffix for this name, default to 1
                names_dict[name] = suffix + 1  # Increment suffix for this name and store it in the dictionary
                if suffix > 1:
                    name = f"{name}{suffix}"  # Add suffix to the name if necessary]
        
                current_hurricane = {'name': name, 'rows': []} # store the name and an empty list for the rows
                hurricane_info.append(current_hurricane)
            # if this is a row of data for the current hurricane
            elif current_hurricane is not None:
                row_data = line.strip().split(',')
                row = {col: row_data[i].strip() for i, col in enumerate(hurricane_columns)}
                current_hurricane['rows'].append(row)

    # Create a new DataFrame from the rows for each hurricane
    final = []
    for hurricane in hurricane_info:
        for row in hurricane['rows']:
            row['name'] = hurricane['name']
            final.append(row)
    processed_df = pd.DataFrame(final)
    processed_df = processed_df[['date', 'name', 'latitude', 'longitude', 'max_wind', 'min_pressure']]
    processed_df = preProcess_weather_data(processed_df)
    return processed_df

        

    ########################################### General #################################################

    """
This calculates the depth of the water given <lat, lon> values.
This done using the sea_depth.tiff file.
Extracted from:
NOAA https://www.ncei.noaa.gov/maps/grid-extract/
"""
def get_sea_depth(latitude, longitude):
    file_path = 'C:/Users/Ben/Documents/GitHub/Animal-Motion-for-predictions/data/Sea_Depth_tiffs/sea_depth.tiff'
    try:
        with rasterio.open(file_path) as dataset:
            row, col = dataset.index(longitude, latitude)
            sea_depth = dataset.read(1, window=((row, row+1), (col, col+1)))
            return sea_depth[0][0]
    
    except (rasterio.errors.RasterioIOError, IndexError) as e:
        # print(f"Error: {e}")
        return None
    
"""
makes a dictionary based in the g_p datasets which combines both weather and animal data. 
"""
def make_cp_dict(g_p, a_data, full_hurricane_1992):
    grr = g_p.groupby('event_name')
    Crossed_points = {}
    for event_na in g_p['event_name'].unique():
        event_data = full_hurricane_1992.loc[full_hurricane_1992['name']== event_na]
        curernt_event_group = grr.get_group(event_na)
        event_start_date = event_data['date'].min()
        event_end_date = event_data['date'].max()
        
        Crossed_points[event_na] = {
                                    'event name': event_na,
                                    'start date': event_start_date,
                                    'end date': event_end_date,
                                    'event data': event_data,
                                    'animals':  {}
                                    }
            
        for animal_na in curernt_event_group['name'].unique():
            animal_data = a_data[a_data['name']==animal_na]
            animal_data = animal_data.sort_values('date')
            animal_before = animal_data.loc[(animal_data['date'] < event_start_date) & (animal_data['date'] >= event_start_date - pd.Timedelta(days=7))]
            animal_during = animal_data.loc[(animal_data['date'] >= event_start_date) & (animal_data['date'] <= event_end_date)]
            animal_after = animal_data.loc[(animal_data['date'] > event_end_date) & (animal_data['date']< event_end_date + pd.Timedelta(days=7))]
            animal_dict = {
                            'animal name': animal_na,
                            'animal data': animal_data,
                            'before event': animal_before,
                            'during event': animal_during,
                            'after event': animal_after,
                        }
            Crossed_points[event_na]['animals'][animal_na] = animal_dict
            
    return Crossed_points

"""
Functions to preprocess for classifiers.

"""
def preprocess_4_LR(path):

    t = pd.read_csv(path)

    t['Barometric Pressure'] = t['Barometric Pressure'].replace('-', float('nan'))
    t['Wind Speed (kn)'] = t['Wind Speed (kn)'].replace('-', float('nan'))
    t['Air Temp (°F)'] = t['Air Temp (°F)'].replace('-', float('nan'))
    t['Wind Gust (kn)'] = t['Wind Gust (kn)'].replace('-', float('nan'))
    t['Humidity (%)'] = t['Humidity (%)'].replace('-', float('nan'))

    t['Barometric Pressure'] = t['Barometric Pressure'].astype(float)
    t['Wind Speed (kn)'] = t['Wind Speed (kn)'].astype(float)
    t['Air Temp (°F)'] = t['Air Temp (°F)'].astype(float)
    t['Wind Gust (kn)'] = t['Wind Gust (kn)'].astype(float)
    t['Humidity (%)'] = t['Humidity (%)'].astype(float)
    
    return t






