import requests
import json

while(True):
    p1 = requests.get('http://192.168.0.150:3000/cmd/2').text
    jsonObject1 = json.loads(p1)
    jsonArray1= jsonObject1.get("user")
    print(jsonArray1)
