import requests
from predict import predict_endpoint 

ride = {
    "PULocationID": 10,
    "DOLocationID": 50,
    "trip_distance": 40
}

#url = 'http://localhost:9696/predict'
#response = requests.post(url, json=ride)
#print(response.json())

print(predict_endpoint(ride))