import requests

url = "http://localhost:5123/process"
files = {"image": open("./test06.jpg", "rb")}
data = {"threshold": "170", "invert": "true"}  # Optional parameters

response = requests.post(url, files=files, data=data)

if response.status_code == 200:
    with open("./output1.png", "wb") as f:
        f.write(response.content)
    print("Image processed successfully!")
else:
    print(f"Error: {response.text}")