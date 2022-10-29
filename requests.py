import requests

url = 'http://127.0.0.1:5000/predict'
r = requests.post(url,json={'experience':2})

print(r.json())