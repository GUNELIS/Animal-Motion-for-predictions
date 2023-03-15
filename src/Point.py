class Point:

    def __init__(self, longitude, latitude, date, Name):
        self.longitude = longitude
        self.latitude = latitude
        self.date = date

    def __str__(self):
        return f"({self.longitude}, {self.latitude}, {self.date})"
    