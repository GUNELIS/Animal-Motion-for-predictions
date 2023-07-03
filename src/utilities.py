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
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier



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
            return True,weather_data['name'].iloc[0],days_from_event, distance_from_event
    else:
        return False,'Nothing',100000,100000


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


def get_cross_points2(all_hurricanes,animals,max_dist):
    """
    This function finds all the points where an animal and a hurricane where
    both found in the same area at the same time. 
    Parameters: dataframe containing the hurricanes and a dataframe containing the animal locations
    Returns: a data frame with all animals in the same place and time as a weather event
    """
    #Defining the new data frame to be returned
    good_pnts = pd.DataFrame(columns = ['latitude',
                                        'longitude',
                                        'date',
                                        'name',
                                        'species',
                                        'depth',
                                        'Dist from Feeding Spot (km)',
                                        'event_name',
                                        'day_dif_from_event',
                                        'distance_from_event_in_km',
                                        'Barometric Pressure',
                                        'Wind Speed (kn)',
                                        'Wind Gust (kn)',
                                        'Air Temp (°F)',
                                        'Humidity (%)',
                                        'severe weather event'
                                        ])
    grouped_hurricanes = all_hurricanes.groupby(all_hurricanes['name'])
    
    for name, group in grouped_hurricanes:
        hurricane = grouped_hurricanes.get_group(name)
        # hurricane = preProcess_weather_data(hurricane)
        for i in range(animals.shape[0]):
            lat = animals['latitude'].iloc[i]
            lon = animals['longitude'].iloc[i]
            date = animals['date'].iloc[i]
            id = animals['name'].iloc[i]
            species = animals['species'].iloc[i]
            swim_depth = animals['depth'].iloc[i]
            distance_from_feeding = animals['Dist from Feeding Spot (km)'].iloc[i]
            baro = animals['Barometric Pressure'].iloc[i]
            wind_speed = animals['Wind Speed (kn)'].iloc[i]
            wind_gust = animals['Wind Gust (kn)'].iloc[i]
            air_temp = animals['Air Temp (°F)'].iloc[i]
            humidity = animals['Humidity (%)'].iloc[i]
            is_event = animals['severe weather event'].iloc[i]
            # is_rp_rt,relevant_event,days_from_event,distance_from_event = right_time_right_place(hurricane,lat,lon,date,400,10)
            rp_rt_vals  = right_time_right_place(hurricane,lat,lon,date,max_dist,day_frame = 10)
            if rp_rt_vals is not None:
                is_rp_rt,relevant_event,days_from_event,distance_from_event = rp_rt_vals
                if is_rp_rt == True:
                    row = {'latitude': lat, 
                        'longitude':lon,
                        'date': date, 
                        'name':id, 
                        'species':species,
                        'depth':swim_depth,
                        'dist from feeding spot':distance_from_feeding,
                        'event_name': relevant_event, 
                        'day_dif_from_event': days_from_event, 
                        'distance_from_event_in_km' : distance_from_event,
                        'Barometric Pressure':baro,
                        'Wind Speed (kn)': wind_speed,
                        'Wind Gust (kn)': wind_gust,
                        'Air Temp (°F)': air_temp,
                        'Humidity (%)': humidity,
                        'severe weather event': is_event
                    }
                
                    good_pnts = good_pnts.append(row, ignore_index= True)
        print('Finished examining hurricane', name)
    return good_pnts  


def get_hurricane_dates(hurricane_data,hurricane_name):
    """
    Takes in the all hurricanes since 1992 and the name of a specific hurricane
    Returns the start and end date of the weather event.  
    """
    start_date = hurricane_data[hurricane_data['name']==hurricane_name]['date'].min()
    end_date = hurricane_data[hurricane_data['name']==hurricane_name]['date'].max() 
    return start_date, end_date	


def get_timecrossed_points(all_hurricanes,animals,dayFrame):
    
    """
    This function returns all the points which were in the same time as the hurricane.
    """
    
    good_pnts = pd.DataFrame(columns = ['latitude','longitude','date','name','event_name','day_dif_from_event','distance_from_event_in_km'])

    grouped_hurricanes = all_hurricanes.groupby(all_hurricanes['name'])
    
    for name, group in grouped_hurricanes:
        hurricane = grouped_hurricanes.get_group(name)
        IDA_mean_lat = hurricane['latitude'].mean()
        IDA_mean_lon = hurricane['longitude'].mean()
        IDA_end = hurricane['date'].max()
        IDA_start = hurricane['date'].min()
        # hurricane = preProcess_weather_data(hurricane)
        for i in range(animals.shape[0]):
            lat = animals['latitude'].iloc[i]
            lon = animals['longitude'].iloc[i]
            date = animals['date'].iloc[i]
            id = animals['name'].iloc[i]
            species = animals['species'].iloc[i]
            
            distance_from_event, within_distance = haversine2(hurricane,lat,lon,100000)
            in_frame, days_from_event = within_time_frame(IDA_start,IDA_end,date,dayFrame) # Time frame
            if in_frame == True:
                row = {'latitude': lat, 
                    'longitude':lon,
                    'date': date, 
                    'name':id, 
                    'species':species, 
                    'event_name': name, 
                    'day_dif_from_event': days_from_event, 
                    'distance_from_event_in_km' : distance_from_event}
                
                good_pnts = good_pnts.append(row, ignore_index= True)
        print('Finished examining hurricane', name, ' there are ',len(good_pnts),' points in g_p' )
    return good_pnts


def get_LR_report(animal):
    
    """
        This function takes in a telemtry dataset and returns a report specifically for logistic regression
        it prints evaluation metrics such as 'Accuracy', 'Precision', 'Recall', 'F1-Score' 
        It also considers the confidence interval.
    """

    selected_columns = ['longitude', 'latitude', 'year', 'depth',
                        'baro pressure', 'wind speed', 'Air Temp (°F)',
                        'Dist from Feeding Spot (km)', 'severe weather event']

    selected_df = animal[selected_columns]
    selected_df.dropna(inplace=True)
    X = selected_df.iloc[:, :-1]
    y = selected_df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=53)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    n = len(y_test)
    z = stats.norm.ppf((1 + 0.95) / 2)

    accuracy_interval = z * np.sqrt((accuracy * (1 - accuracy)) / n)
    precision_interval = z * np.sqrt((precision * (1 - precision)) / n)
    recall_interval = z * np.sqrt((recall * (1 - recall)) / n)
    f1_interval = z * np.sqrt((f1 * (1 - f1)) / n)

    evaluation_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Score': [accuracy, precision, recall, f1],
        'Confidence Interval': [accuracy_interval, precision_interval, recall_interval, f1_interval]
    })

    print("Evaluation Report:")
    print(evaluation_df)





def get_report(animal):

    """
        
        This function takes in a telemtry dataset and returns a report  for all classifiers. 
        it prints evaluation metrics such as 'Accuracy', 'Precision', 'Recall', 'F1-Score' 
        It also considers the confidence interval.
    
    """


    selected_columns = ['longitude', 'latitude', 'year', 'depth',
                        'baro pressure', 'wind speed', 'Air Temp (°F)',
                        'Dist from Feeding Spot (km)', 'severe weather event']

    selected_df = animal[selected_columns]
    selected_df.dropna(inplace=True)
    X = selected_df.iloc[:, :-1]
    y = selected_df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=53)
    classifiers = [
        LogisticRegression(),
        SVC(kernel='linear'),
        RandomForestClassifier(),
        GradientBoostingClassifier()
    ]

    results = {}

    # Iterate over the classifiers
    for classifier in classifiers:
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        confidence = 0.95
        n = len(y_test)
        z = stats.norm.ppf((1 + confidence) / 2)

        accuracy_interval = z * np.sqrt((accuracy * (1 - accuracy)) / n)
        precision_interval = z * np.sqrt((precision * (1 - precision)) / n)
        recall_interval = z * np.sqrt((recall * (1 - recall)) / n)
        f1_interval = z * np.sqrt((f1 * (1 - f1)) / n)

        results[type(classifier).__name__] = {
            'Accuracy': accuracy,
            'Accuracy Interval': accuracy_interval,
            'Precision': precision,
            'Precision Interval': precision_interval,
            'Recall': recall,
            'Recall Interval': recall_interval,
            'F1-Score': f1,
            'F1-Score Interval': f1_interval
        }

    evaluation_df = pd.DataFrame.from_dict(results, orient='index')

    print("Evaluation Report:")
    print(evaluation_df)
    return evaluation_df
