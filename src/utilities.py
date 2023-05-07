import Point
import datetime
import pandas as pd
import folium
from IPython.display import display
from math import radians, cos, sin, asin, sqrt


# checks if a point is in a certain area
# Input: 
# 1 - the point in question (the animal's location)
# 2 - the center of the area we are checking
# 3 - buffer, the length of the area
# Output: True if in area


color_list =  ['beige', 'black', 'blue', 'cadetblue', 'darkblue', 'darkgreen', 'darkpurple', 'darkred', 'gray', 'green', 'lightblue', 'lightgray', 'lightgreen', 'lightred', 'orange', 'pink', 'purple', 'red', 'white']

def is_in_area(point2check: Point.Point, central_point: Point.Point, buffer: float):
    
    lat_dist = abs(point2check.latitude - central_point.latitude)
    long_dist = abs(point2check.longitude - central_point.longitude)
    if long_dist <= buffer and lat_dist <= buffer:
        return True
    return False

def right_time_right_place(weather_data,lat2,lon2,event,max_dist=260,day_frame=20):

    """
    This functions checks whether an animal was present in the same time and place as the 
    weather event.
    First the mean loction of the hurrican/tornado is taken as well as start and end date. 
    """
    IDA_mean_lat = weather_data['latitude'].mean()
    IDA_mean_lon = weather_data['longitude'].mean()
    IDA_end = weather_data['date'].max()
    IDA_start = weather_data['date'].min()

    in_frame, days_from_event = within_time_frame(IDA_start,IDA_end,event,day_frame) # Time frame
    if in_frame:
        distance_from_event, within_distance = haversine2(weather_data,lat2,lon2,max_dist)
        if within_distance:
            print('distnace max is ', max_dist, ' and dist is ', distance_from_event)
            # if weather_data['name'].iloc[0] == None or days_from_event == None or distance_from_event== None:
            print('------------------ben==   ----  ',weather_data['name'].iloc[0],days_from_event, distance_from_event)
            return True,weather_data['name'].iloc[0],days_from_event, distance_from_event
    else:
        return False,'Nothing',100000,100000
    # distance_from_event = haversine(IDA_mean_lat,IDA_mean_lon,lat2,lon2)             # Distance
    # if  distance_from_event < max_dist and in_frame:
    #     return True,weather_data['name'].iloc[0],days_from_event, distance_from_event
    # else:
    #     return False,'Nothing',100000,100000


# takes the csv and converts each row in the df to a point object
def transform_to_points(path):
    df = pd.read_csv(path)
    points = [Point.Point(row['Longitude'], row['Latitude'], row['Date'], row['Event']) for i, row in df.iterrows()]
    return points


def within_time_frame(start,end,event,dayFrame):
    """
    Checks whether event1 occured within n days from event2.
    returns True if difference is smaller than dayFrame.
    event1, event2 - DateTime variables.
    dayFrame - any positive integer.  
    """
    if dayFrame<0:
        print('Only positive values allowed')
        return False
    
    # if event <= end+datetime.timedelta(days=dayFrame) and event >= str(datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')-datetime.timedelta(days=dayFrame)) :
    if event <= end+datetime.timedelta(days=dayFrame) and event >= start-datetime.timedelta(days=dayFrame) :
        days = min((abs(event-end)), abs(event-start)) # get the difference in days from the event
        return True, days
    else:
        return False, 100000

# Inspired from https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points

def haversine2(hurricane, lat2, lon2, dist_limit):
    """
    Function that uses the Haversine formula to calculate whether a single point 
    Is within a certain distance from any of the points in the hurricane list.
    Parameters: 
    Hurricane - list of coordinates, <lat2,lon2> - the coordinates of the animal
    Returns true if the animal is within the given distance from any point of the hurricane.
    """
    for lon1, lat1 in zip(hurricane['longitude'],hurricane['latitude']):
    # degrees to radians conversion 
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        # haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        dist = c * 6371 # Radius of earth in kilometers
        if dist <= dist_limit:
            return dist, True
    return 10000, False

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the Earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return c * 6371 # Radius of earth in kilometers

def get_quick_marked_map(data, the_icon ='wind-turbine',color='blue'):
    mean_lat = data['latitude'].mean()
    mean_lon = data['longitude'].mean()
    map = folium.Map(location=[mean_lat, mean_lon], zoom_start=4)

    for lat,lon in zip(data['latitude'], data ['longitude']):

        folium.Marker([lat, lon], icon=folium.Icon(color=color,icon=the_icon,prefix='fa')).add_to(map)
    
    return map

def mark_map(map,data, the_icon ='lion',color='blue'):
    mean_lat = data['latitude'].mean()
    mean_lon = data['longitude'].mean()
    print(data.columns)
    for i in range(data.shape[0]):
        lat = data['latitude'].iloc[i]
        lon = data['longitude'].iloc[i]
        date2 = data['date'].iloc[i]
        name = data['name'].iloc[i]

    # for lat,lon in zip(data['latitude'], data ['longitude']):
        
        folium.Marker([lat, lon], popup = [name, date2], icon=folium.Icon(color=color,icon=the_icon,prefix='fa')).add_to(map)
    
    return map

def preProcess_weather_data(data):
    """
    The weather data contains coordinates in the format of N and W. eg: 32.43N and 23.3W
    This function makes all N & E positive and S & W negative 
    """
    
    if isinstance(data['date'].iloc[0], str):
        print('ýessssssssssssssssssssssssssss')
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


def get_cross_points(all_hurricanes,animals,max_dist):
    """
    This function finds all the points where an animal and a hurricane where
    both found in the same area at the same time. 
    Parameters: dataframe containing the hurricanes and a dataframe containing the animal locations
    Returns: a data frame with all animals in the same place and time as a weather event
    """
    #Defining the new data frame to be returned
    good_pnts = pd.DataFrame(columns = ['latitude','longitude','date','name','event_name','day_dif_from_event','distance_from_event_in_km'])

    grouped_hurricanes = all_hurricanes.groupby(all_hurricanes['name'])
    
    for name, group in grouped_hurricanes:
        hurricane = grouped_hurricanes.get_group(name)
        # hurricane = preProcess_weather_data(hurricane)
        for i in range(animals.shape[0]):
            lat = animals['latitude'].iloc[i]
            lon = animals['longitude'].iloc[i]
            date = animals['date'].iloc[i]
            id = animals['name'].iloc[i]
            # is_rp_rt,relevant_event,days_from_event,distance_from_event = right_time_right_place(hurricane,lat,lon,date,400,10)
            rp_rt_vals  = right_time_right_place(hurricane,lat,lon,date,max_dist = 400,day_frame = 10)
            if rp_rt_vals is not None:
                is_rp_rt,relevant_event,days_from_event,distance_from_event = rp_rt_vals
                if is_rp_rt == True:
                    row = {'latitude': lat, 
                        'longitude':lon,
                        'date': date, 
                        'name':id, 
                        'event_name': relevant_event, 
                        'day_dif_from_event': days_from_event, 
                        'distance_from_event_in_km' : distance_from_event}
                
                    good_pnts = good_pnts.append(row, ignore_index= True)
    return good_pnts       

def printben():
    print('hey ben')



def blue_whale_preprocess(path):
    blue_whales = pd.read_csv(path)
    # temp_df = blue_whales['event-id'].str.split(',', expand=True)
    # temp_df.columns = ['event-id', 'visible', 'timestamp', 'location-long', 'location-lat',
    #    'argos:best-level', 'argos:calcul-freq', 'argos:iq','argos:lat1','argos:lat2','argos:lc','argos:lon1','argos:lon2','argos:nb-mes','argos:nb-mes-120','manually-marked-outlier','sensor-type','individual-taxon-canonical-name','tag-local-identifier','individual-local-identifier','study-name']
    
    # temp_df = temp_df.rename(columns={'location-long': 'longitude', 'location-lat': 'latitude', 'tag-local-identifier': 'name'})
    # blue_whales =temp_df.iloc[:,[0,1,2,3,4,18]].copy() # refine the csv and get only the desired columns from og dataset
    blue_whales =blue_whales.iloc[:,[0,1,2,3,4,18]].copy() # refine the csv and get only the desired columns from og dataset
    blue_whales = blue_whales.rename(columns={'location-long': 'longitude', 'location-lat': 'latitude', 'tag-local-identifier': 'name'})

    # sperm_whales.to_csv('Sperm_whales_refined.csv', index=False)

    # A bit more of data preprocessing, making all coordinate values numeric
    for i, row in blue_whales.iterrows():
        blue_whales.at[i, "longitude"] = float(row["longitude"])
        blue_whales.at[i, "latitude"] = float(row["latitude"])

    # More data preprocessing, getting the timestamp column ready

    blue_whales = blue_whales.rename(columns={'timestamp': 'date'})
    blue_whales = blue_whales.rename(columns={'individual-local-identifier': 'name'})
    blue_whales['date'] = pd.to_datetime(blue_whales['date'])

    return blue_whales



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



# # Hurricane_Katrina = Point.Point(-89.4075, 30.28669 ,datetime.date(2005, 8, 23))
# # animal_location = Point.Point(-80.4075, 20.2669, None)
# # print(is_in_area(animal_location,Hurricane_Katrina, 10.01))
# print('ok starting./..')
# points = transform_to_points('data/disasters.csv')
# for p  in points: print(p)