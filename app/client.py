import requests

response = requests.post(
    "http://localhost:8008/generate",
    json={"prompt": "Hello GPT"}
)
print(response.json())