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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve
from sklearn.inspection import permutation_importance


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
"""
Standard folium map marking given the chosen icon , color and locations

"""
def mark_map(map,data, the_icon ='lion',color='blue'):
    mean_lat = data['latitude'].mean()
    mean_lon = data['longitude'].mean()
    for i in range(data.shape[0]):
        lat = data['latitude'].iloc[i]
        lon = data['longitude'].iloc[i]
        date2 = data['date'].iloc[i]
        name = data['name'].iloc[i]        
        folium.Marker([lat, lon], popup = [name, date2], icon=folium.Icon(color=color,icon=the_icon,prefix='fa')).add_to(map)
    MousePosition().add_to(map)
    return map

"""
Marking the points but unpacked from the dictionary.

"""
def mark_dict_map(map,dict):
    for hurricane_name in dict.keys():
        group = folium.FeatureGroup(name = hurricane_name).add_to(map)
        event_data = dict[hurricane_name]['event data']
        for i in range(event_data.shape[0]):
                lat = event_data['latitude'].iloc[i]
                lon = event_data['longitude'].iloc[i]
                date2 = event_data['date'].iloc[i]
                folium.Marker([lat, lon], popup = [hurricane_name, date2], icon=folium.Icon(color='pink',icon='tornado',prefix='fa')).add_to(group)
        for animal_name in dict[hurricane_name]['animals'].keys():
            animal_data = dict[hurricane_name]['animals'][animal_name]['animal data']
            # group_before = folium.FeatureGroup(name = f"{animal_name} before event").add_to(group)
            # line_before = folium.PolyLine(locations=list(zip(before['latitude'],before['longitude'])),color='blue').add_to(group_before)
            before = dict[hurricane_name]['animals'][animal_name]['before event']
            during = dict[hurricane_name]['animals'][animal_name]['during event']
            after = dict[hurricane_name]['animals'][animal_name]['after event']

            # mark_map(group_before,before,'fish','blue').add_to(group_before)
            for i in range(animal_data.shape[0]):
                lat = animal_data['latitude'].iloc[i]
                lon = animal_data['longitude'].iloc[i]
                date2 = animal_data['date'].iloc[i]
                if  is_in(before['date'],date2):
                    folium.Marker([lat, lon], popup = [animal_name, date2], icon=folium.Icon(color='blue',icon='fish',prefix='fa')).add_to(group)
                if is_in(during['date'],date2):
                    folium.Marker([lat, lon], popup = [animal_name, date2], icon=folium.Icon(color='green',icon='fish',prefix='fa')).add_to(group)
                if is_in(after['date'],date2):
                    folium.Marker([lat, lon], popup = [animal_name, date2], icon=folium.Icon(color='red',icon='fish',prefix='fa')).add_to(group)
                else:
                    folium.Marker([lat, lon], popup = [animal_name, date2], icon=folium.Icon(color='orange',icon='dog',prefix='fa')).add_to(group)

            # folium.Marker([lat, lon], popup = [name, date2], icon=folium.Icon(color=color,icon=the_icon,prefix='fa')).add_to(group)
    return map

def mark_layered_map(map,h_name,data,the_icon ='lion',color='blue'):
    group = folium.FeatureGroup(name = h_name)
    mean_lat = data['latitude'].mean()
    mean_lon = data['longitude'].mean()
    for i in range(data.shape[0]):
        lat = data['latitude'].iloc[i]
        lon = data['longitude'].iloc[i]
        date2 = data['date'].iloc[i]
        name = data['name'].iloc[i]        
        folium.Marker([lat, lon], popup = [name, date2], icon=folium.Icon(color=color,icon=the_icon,prefix='fa')).add_to(group)
    group.add_to(map)
 
    MousePosition().add_to(map)

    return map


"""
Automated way to show the before during after of a certain individual corresponding to a certain hurricane
"""
def mark_before_during_after(map,name,before,during,after):
    if not before.empty:
        group_before = folium.FeatureGroup(name = f"Indidvidual {name} Trajectory before event").add_to(map)
        # line_before = folium.PolyLine(locations=list(zip(before['latitude'],before['longitude'])),color='blue').add_to(group_before)
        mark_map(group_before,before,'fish','blue')
        hm_group_before = folium.FeatureGroup(name = f"Indidvidual {name} HeatMap before event").add_to(map)
        heatmap_before = plugins.HeatMap(data=before[['latitude', 'longitude']], radius=15).add_to(hm_group_before)

    if not during.empty:
        group_during = folium.FeatureGroup(name = f"Indidvidual {name} Trajectory during event").add_to(map)
        # line_during = folium.PolyLine(locations=list(zip(during['latitude'],during['longitude'])),color='green').add_to(group_during)
        mark_map(group_during,during,'fish','green')
        hm_group_during = folium.FeatureGroup(name = f"Indidvidual {name} HeatMap during event").add_to(map)
        heatmap_during = plugins.HeatMap(data=during[['latitude', 'longitude']], radius=15).add_to(hm_group_during)

    if not after.empty:    
        group_after = folium.FeatureGroup(name =  f"Indidvidual {name} Trajectory after event").add_to(map)
        # line_after = folium.PolyLine(locations=list(zip(after['latitude'],after['longitude'])),color='red').add_to(group_after)
        mark_map(group_after,after,'fish','red')
        hm_group_after = folium.FeatureGroup(name =  f"Indidvidual {name} HeatMap after event").add_to(map)
        heatmap_after = plugins.HeatMap(data=after[['latitude', 'longitude']], radius=15).add_to(hm_group_after)
    
    add_sea_depth(map) # Adds an image of the depth of the see based on 4 coordinates

    return map

"""
Overlays the  sea depth raster on the original map.

"""
def add_sea_depth(map):
    sea_depth_img = 'C:/Users/Ben/Documents/GitHub/Animal-Motion-for-predictions/data/Images/depth.png'
    map_group = folium.FeatureGroup(name='Sea Depth Golf of Mexico').add_to(map)
    image_overlay = folium.raster_layers.ImageOverlay(
        image= sea_depth_img,
        bounds=[[30.8936, -100.5469], [16.3037, -71.3672]],
        opacity=0.5)
    image_overlay.add_to(map_group)

"""
Creates a heat map given the map and animal location data
"""
def make_heatMap(map, animals):
    species = animals['species'].iloc[0]
    hm_group = folium.FeatureGroup(name = f"Heat Map of {species}").add_to(map)
    heatmap_during = plugins.HeatMap(data=animals[['latitude', 'longitude']], radius=15).add_to(hm_group)
    MousePosition().add_to(map)
    return map

"""
Creates a polyline representation of shark trajectories
"""
def make_polylines(map, animals):
    i = len(color_list)-1
    for shark_name, group in animals.groupby('name'):
        feature_group = folium.FeatureGroup(name = shark_name).add_to(map)
        line = folium.PolyLine(locations = list(zip(group['latitude'],group['longitude'])), color = color_list[i]).add_to(feature_group)
        i = i-1
        if i==-1:
            i = len(color_list)-1
    folium.map.LayerControl(position='topright', collapsed=False).add_to(map)
    MousePosition().add_to(map)
    return map

"""
Marks multiple values in one map
"""
def mark_map_multiple(map,animal, the_icon ='lion'):
    color_index = len(color_list)-1
    for shark_name, group in animal.groupby('name'):
        feature_group = folium.FeatureGroup(name = shark_name).add_to(map)
        mean_lat = group['latitude'].mean()
        mean_lon = group['longitude'].mean()
        for i in range(group.shape[0]):
            lat = group['latitude'].iloc[i]
            lon = group['longitude'].iloc[i]
            date2 = group['date'].iloc[i]
            name = group['name'].iloc[i]        
            folium.Marker([lat, lon], popup = [name, date2], icon=folium.Icon(color=color_list[color_index],icon=the_icon,prefix='fa')).add_to(feature_group)
            color_index = color_index-1
            if color_index==-1:
                color_index = len(color_list)-1
    folium.map.LayerControl(position='topright', collapsed=False).add_to(map)
    MousePosition().add_to(map)
    return map

"""
Plotting the model performance in several subplots
"""
def plot_model_Evaluation(model,y_test,y_pred,X_test,X):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axs[0, 0])
    axs[0, 0].set_title("Confusion Matrix")
    axs[0, 0].set_xlabel("Predicted Labels")
    axs[0, 0].set_ylabel("True Labels")

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    roc_auc = roc_auc_score(y_test, y_pred)

    axs[0, 1].plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    axs[0, 1].plot([0, 1], [0, 1], "k--")
    axs[0, 1].set_xlim([0.0, 1.0])
    axs[0, 1].set_ylim([0.0, 1.05])
    axs[0, 1].set_xlabel("False Positive Rate")
    axs[0, 1].set_ylabel("True Positive Rate")
    axs[0, 1].set_title("Receiver Operating Characteristic")
    axs[0, 1].legend(loc="lower right")

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

    axs[1, 0].plot(recall, precision)
    axs[1, 0].set_xlim([0.0, 1.0])
    axs[1, 0].set_ylim([0.0, 1.05])
    axs[1, 0].set_xlabel("Recall")
    axs[1, 0].set_ylabel("Precision")
    axs[1, 0].set_title("Precision-Recall Curve")

    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    sorted_idx = result.importances_mean.argsort()

    axs[1, 1].barh(X.columns[sorted_idx], result.importances_mean[sorted_idx])
    axs[1, 1].set_xlabel("Feature Importance")
    axs[1, 1].set_title("Feature Importance")

    plt.tight_layout()

    plt.show()


def is_in(dates, date):
    for d in dates:
        if d.date() == date.date():
            return True
    return False
