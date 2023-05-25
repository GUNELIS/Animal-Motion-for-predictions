import Point
import datetime
import pandas as pd
import folium
from IPython.display import display
from math import radians, cos, sin, asin, sqrt
import folium.plugins as plugins
from folium.plugins import MousePosition
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy import stats



# checks if a point is in a certain area
# Input: 
# 1 - the point in question (the animal's location)
# 2 - the center of the area we are checking
# 3 - buffer, the length of the area
# Output: True if in area


color_list =  ['cadetblue','beige', 'darkred','black', 'orange','darkpurple', 'pink', 'blue', 'darkblue', 'darkgreen', 'gray', 'green', 'lightblue', 'lightgray', 'lightgreen', 'lightred', 'purple', 'red', 'white']
freq_visited_coordinates = [
    (27.07892, -79.231),
    (24.967, -80.940),
    (24.52352, -82.78676),
    (26.20509, -82.05591),
    (27.90888, -84.174475),
    (29.2778, -84.163),
    (30.32088, -86.5547),
    (28.06455, -94.51558),
    (28.36448, -91.68008)
]
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


# def mark_map(map,data, the_icon ='lion',color='blue'):
#     mean_lat = data['latitude'].mean()
#     mean_lon = data['longitude'].mean()
#     for i in range(data.shape[0]):
#         lat = data['latitude'].iloc[i]
#         lon = data['longitude'].iloc[i]
#         date2 = data['date'].iloc[i]
#         name = data['name'].iloc[i]        
#         folium.Marker([lat, lon], popup = [name, date2], icon=folium.Icon(color=color,icon=the_icon,prefix='fa')).add_to(map)
#     MousePosition().add_to(map)
#     return map


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
        print('Finished examining hurricane', name)
    return good_pnts   


# def mark_layered_map(map,h_name,data,the_icon ='lion',color='blue'):
#     group = folium.FeatureGroup(name = h_name)
#     mean_lat = data['latitude'].mean()
#     mean_lon = data['longitude'].mean()
#     for i in range(data.shape[0]):
#         lat = data['latitude'].iloc[i]
#         lon = data['longitude'].iloc[i]
#         date2 = data['date'].iloc[i]
#         name = data['name'].iloc[i]        
#         folium.Marker([lat, lon], popup = [name, date2], icon=folium.Icon(color=color,icon=the_icon,prefix='fa')).add_to(group)
#     group.add_to(map)
 
#     MousePosition().add_to(map)

#     return map

# def mark_before_during_after(map,name,before,during,after):
#     if not before.empty:
#         group_before = folium.FeatureGroup(name = f"Indidvidual {name} Trajectory before event").add_to(map)
#         # line_before = folium.PolyLine(locations=list(zip(before['latitude'],before['longitude'])),color='blue').add_to(group_before)
#         mark_map(group_before,before,'fish','blue')
#         hm_group_before = folium.FeatureGroup(name = f"Indidvidual {name} HeatMap before event").add_to(map)
#         heatmap_before = plugins.HeatMap(data=before[['latitude', 'longitude']], radius=15).add_to(hm_group_before)

#     if not during.empty:
#         group_during = folium.FeatureGroup(name = f"Indidvidual {name} Trajectory during event").add_to(map)
#         # line_during = folium.PolyLine(locations=list(zip(during['latitude'],during['longitude'])),color='green').add_to(group_during)
#         mark_map(group_during,during,'fish','green')
#         hm_group_during = folium.FeatureGroup(name = f"Indidvidual {name} HeatMap during event").add_to(map)
#         heatmap_during = plugins.HeatMap(data=during[['latitude', 'longitude']], radius=15).add_to(hm_group_during)

#     if not after.empty:    
#         group_after = folium.FeatureGroup(name =  f"Indidvidual {name} Trajectory after event").add_to(map)
#         # line_after = folium.PolyLine(locations=list(zip(after['latitude'],after['longitude'])),color='red').add_to(group_after)
#         mark_map(group_after,after,'fish','red')
#         hm_group_after = folium.FeatureGroup(name =  f"Indidvidual {name} HeatMap after event").add_to(map)
#         heatmap_after = plugins.HeatMap(data=after[['latitude', 'longitude']], radius=15).add_to(hm_group_after)
    
#     add_sea_depth(map) # Adds an image of the depth of the see based on 4 coordinates
#     folium.map.LayerControl(position='topright', collapsed=False).add_to(map)
#     MousePosition().add_to(map)

#     return map

# def make_heatMap(map, animals):
#     species = animals['species'].iloc[0]
#     hm_group = folium.FeatureGroup(name = f"Heat Map of {species}").add_to(map)
#     heatmap_during = plugins.HeatMap(data=animals[['latitude', 'longitude']], radius=15).add_to(hm_group)
#     MousePosition().add_to(map)
#     return map


# def make_polylines(map, animals):
#     i = len(color_list)-1
#     for shark_name, group in animals.groupby('name'):
#         feature_group = folium.FeatureGroup(name = shark_name).add_to(map)
#         line = folium.PolyLine(locations = list(zip(group['latitude'],group['longitude'])), color = color_list[i]).add_to(feature_group)
#         i = i-1
#         if i==-1:
#             i = len(color_list)-1
#     folium.map.LayerControl(position='topright', collapsed=False).add_to(map)
#     MousePosition().add_to(map)
#     return map

# def mark_map_multiple(map,animal, the_icon ='lion'):
#     color_index = len(color_list)-1
#     for shark_name, group in animal.groupby('name'):
#         feature_group = folium.FeatureGroup(name = shark_name).add_to(map)
#         mean_lat = group['latitude'].mean()
#         mean_lon = group['longitude'].mean()
#         for i in range(group.shape[0]):
#             lat = group['latitude'].iloc[i]
#             lon = group['longitude'].iloc[i]
#             date2 = group['date'].iloc[i]
#             name = group['name'].iloc[i]        
#             folium.Marker([lat, lon], popup = [name, date2], icon=folium.Icon(color=color_list[color_index],icon=the_icon,prefix='fa')).add_to(feature_group)
#             color_index = color_index-1
#             if color_index==-1:
#                 color_index = len(color_list)-1
#     folium.map.LayerControl(position='topright', collapsed=False).add_to(map)
#     MousePosition().add_to(map)
#     return map


def add_sea_depth(map):
    sea_depth_img = 'C:/Users/Ben/Documents/GitHub/Animal-Motion-for-predictions/data/Images/depth.png'
    map_group = folium.FeatureGroup(name='Sea Depth Golf of Mexico').add_to(map)
    image_overlay = folium.raster_layers.ImageOverlay(
        image= sea_depth_img,
        bounds=[[30.8936, -100.5469], [16.3037, -71.3672]],
        opacity=0.5)
    image_overlay.add_to(map_group)



def get_LR_report(animal):
    
    # Assuming your DataFrame is named "df"
    selected_columns = ['longitude', 'latitude', 'year', 'depth',
                        'Barometric Pressure', 'Wind Speed (kn)', 'Air Temp (°F)',
                        'Dist from Feeding Spot (km)', 'severe weather event']

    # Select the columns of interest from the DataFrame
    selected_df = animal[selected_columns]
    selected_df.dropna(inplace=True)

    # Split the data into independent variables (X) and the dependent variable (y)
    X = selected_df.iloc[:, :-1]
    y = selected_df.iloc[:, -1]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=53)

    # Initialize the logistic regression model
    model = LogisticRegression()

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Compute evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Compute confidence intervals (95% confidence level)
    confidence = 0.95
    n = len(y_test)
    z = stats.norm.ppf((1 + confidence) / 2)

    accuracy_interval = z * np.sqrt((accuracy * (1 - accuracy)) / n)
    precision_interval = z * np.sqrt((precision * (1 - precision)) / n)
    recall_interval = z * np.sqrt((recall * (1 - recall)) / n)
    f1_interval = z * np.sqrt((f1 * (1 - f1)) / n)

    # Create a DataFrame to store the evaluation metrics and confidence intervals
    evaluation_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Score': [accuracy, precision, recall, f1],
        'Confidence Interval': [accuracy_interval, precision_interval, recall_interval, f1_interval]
    })

    # Print the evaluation report with confidence intervals
    print("Evaluation Report:")
    print(evaluation_df)



