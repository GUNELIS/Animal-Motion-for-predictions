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


color_list =  ['cadetblue','darkpurple', 'darkred','pink', 'orange','beige', 'black', 'blue', 'darkblue', 'darkgreen', 'gray', 'green', 'lightblue', 'lightgray', 'lightgreen', 'lightred', 'purple', 'red', 'white']

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
            # print('distnace max is ', max_dist, ' and dist is ', distance_from_event)
            # if weather_data['name'].iloc[0] == None or days_from_event == None or distance_from_event== None:
                # print('------------------ben==   ----  ',weather_data['name'].iloc[0],days_from_event, distance_from_event)
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
        return False,10000
    # if abs(start.year-event['year']) > 1 or abs(end.year - event['year']) > 1:
        # return False,10000 
    # if event <= end+datetime.timedelta(days=dayFrame) and event >= str(datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')-datetime.timedelta(days=dayFrame)) :
    if event <= end and event >= start-datetime.timedelta(days=dayFrame) :
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

def mark_map(map,data, the_icon ='lion',color='blue'):
    mean_lat = data['latitude'].mean()
    mean_lon = data['longitude'].mean()
    for i in range(data.shape[0]):
        lat = data['latitude'].iloc[i]
        lon = data['longitude'].iloc[i]
        date2 = data['date'].iloc[i]
        name = data['name'].iloc[i]        
        folium.Marker([lat, lon], popup = [name, date2], icon=folium.Icon(color=color,icon=the_icon,prefix='fa')).add_to(map)
    
    return map


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
            rp_rt_vals  = right_time_right_place(hurricane,lat,lon,date,max_dist,day_frame = 10)
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

