import Point
import datetime
import pandas as pd

# checks if a point is in a certain area
# Input: 
# 1 - the point in question (the animal's location)
# 2 - the center of the area we are checking
# 3 - buffer, the length of the area
# Output: True if in area

def is_in_area(point2check: Point.Point, central_point: Point.Point, buffer: float):
    
    lat_dist = abs(point2check.latitude - central_point.latitude)
    long_dist = abs(point2check.longitude - central_point.longitude)
    if long_dist <= buffer and lat_dist <= buffer:
        return True
    return False


# takes the csv and converts each row in the df to a point object
def transform_to_points(path):
    df = pd.read_csv(path)
    points = [Point.Point(row['Longitude'], row['Latitude'], row['Date'], row['Event']) for i, row in df.iterrows()]
    return points



# # Hurricane_Katrina = Point.Point(-89.4075, 30.28669 ,datetime.date(2005, 8, 23))
# # animal_location = Point.Point(-80.4075, 20.2669, None)
# # print(is_in_area(animal_location,Hurricane_Katrina, 10.01))
# print('ok starting./..')
# points = transform_to_points('data/disasters.csv')
# for p  in points: print(p)