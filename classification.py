import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("daily_weather.csv")

# print(data.isnull().any(axis = 1))
# print(data[data.isnull().any(axis = 1)])

del data["number"]

data = data.dropna()

data_touse = data.copy()
data_touse["high_hum"] = (data['relative_humidity_3pm'] > 24.99) * 1
y = data_touse[["high_hum"]].copy()

morning_features = ['air_pressure_9am','air_temp_9am','avg_wind_direction_9am','avg_wind_speed_9am',
        'max_wind_direction_9am','max_wind_speed_9am','rain_accumulation_9am',
        'rain_duration_9am']

x = data_touse[morning_features].copy()

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.33, random_state= 324)

humidity_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
humidity_classifier.fit(x_train,y_train)

predictions = humidity_classifier.predict(x_test)
print(predictions[:10])

# print(y_test["high_hum"][:10])

print(accuracy_score(y_true = y_test, y_pred= predictions ))
