import pandas as pd

def sperm_whale_preprocess(path):
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

    # A bit more of data preprocessing, making all coordinate values numeric
    for i, row in sperm_whales.iterrows():
        sperm_whales.at[i, "longitude"] = float(row["longitude"])
        sperm_whales.at[i, "latitude"] = float(row["latitude"])

    # More data preprocessing, getting the timestamp column ready

    sperm_whales = sperm_whales.rename(columns={'timestamp': 'date'})
    sperm_whales = sperm_whales.rename(columns={'individual-local-identifier': 'name'})
    sperm_whales['date'] = pd.to_datetime(sperm_whales['date'])

    return sperm_whales


def blue_whale_preprocess(path):
    blue_whales = pd.read_csv(path)
    blue_whales =blue_whales.iloc[:,[0,1,2,3,4,18]].copy() # refine the csv and get only the desired columns from og dataset
    blue_whales = blue_whales.rename(columns={'location-long': 'longitude', 'location-lat': 'latitude', 'tag-local-identifier': 'name'})

    # A bit more of data preprocessing, making all coordinate values numeric
    for i, row in blue_whales.iterrows():
        blue_whales.at[i, "longitude"] = float(row["longitude"])
        blue_whales.at[i, "latitude"] = float(row["latitude"])

    # More data preprocessing, getting the timestamp column ready

    blue_whales = blue_whales.rename(columns={'timestamp': 'date'})
    blue_whales = blue_whales.rename(columns={'individual-local-identifier': 'name'})
    blue_whales['date'] = pd.to_datetime(blue_whales['date'])

    return blue_whales



def proccess_hurricane_txt(file_path):
    
    """
    This function reads the hurrcane text file taken from the National Hurricane Center nhc.noaa.gov
    The only parameter is the text file path itself and the functions
    returns a pandas data frame with relevant columns in correct form.
    """
    hurricane_columns = ['date', 'Hour UTC', 'RI', 'HU', 'latitude', 'longitude', 'max_wind', 'min_pressure', 'ne34', 'se34', 'sw34', 'nw34', 'ne50', 'se50', 'sw50', 'nw50', 'ne64', 'se64', 'sw64', 'nw64', 'name']
    
    # Read the text file line by line and store the names in a dict of names for easy access later
    hurricane_info = []
    current_hurricane = None
    with open(file_path, 'r') as f:
        for line in f:
            # if this is a new hurricane
            if line.startswith('AL'): # AL is a starter indicating new hurricane information
                name = line.split(',')[1].strip() # get the name which is after the coma
                # row = {'name': name} # store the name
                current_hurricane = name

            else:
                values = line.strip().split(',') # this is information so split it into columns based on the column list
                # row.update(zip(hurricane_columns[:-1], values)) # use all columns besides the last one which is a new column called name
                row = dict(zip(hurricane_columns, values))
                row['name'] = current_hurricane
                hurricane_info.append(row)

    processed_df = pd.DataFrame(hurricane_info, columns= hurricane_columns)
    processed_df =processed_df.iloc[:,[0,1,4,5,6,7,20]].copy()
    processed_df = preProcess_weather_data(processed_df)
    processed_df['year'] = processed_df['date'].dt.year 
    return processed_df


def preProcess_weather_data(data):
    """
    The weather data contains coordinates in the format of N and W. eg: 32.43N and 23.3W
    This function makes all N & E positive and S & W negative 
    """
    
    if isinstance(data['date'].iloc[0], str):
        data['date'] = data['date'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
        
    lat_column = data['latitude']
    lon_column = data['longitude']

    # A loop that checks the latitude
    for i in range(len(lat_column)):
        value = lat_column.iloc[i]
        if "N" in value:
            lat = float(value.strip("N"))
        elif "S" in value:
            lat = -1 * float(value.strip("S"))
        lat_column.iloc[i] = lat
    # A loop that checks the longitude
    for i in range(len(lon_column)):
        value = lon_column.iloc[i]
        if "E" in value:
            lon = float(value.strip("E"))
        elif "W" in value:
            lon = -1 * float(value.strip("W"))
        lon_column.iloc[i] = lon
    data['latitude'] = lat_column
    data['longitude'] = lon_column
    return data




