import json
import time
from typing import List

import requests

token = ("eyJ0eXAiOiJqd3QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VybmFtZSI6ImxjbDE5Mzc5OC0xOTM3OTgiLCJwYXNzd"
         "29yZCI6IjkyMTIwMyIsImV4cCI6MTk5NDkxNTQwNX0.QvFuzu035i5ApMuf1w75vKjEiQ0Dj0QPlfJmuZtfM30")
url = "https://api.mit-spider.alibaba-inc.com/chatgpt/api/ask"
headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}


def chat_completion(prompt: str, n: int = 1, temperature: float = 1.0) -> List[str]:
    assert 0 < temperature < 2
    payload = {
        "model": 'gpt-3.5-turbo',
        "messages": [{"role": "user", "content": prompt}],
        "n": n,
        "temperature": temperature
    }
    payload = json.dumps(payload)
    response = None
    while True:
        try:
            response = requests.request("POST", url, headers=headers, data=payload)
            # response = requests.post(url, json=payload, headers=headers)
            if response.status_code == 200:
                response = json.loads(response.content.decode("utf-8"))
                if response['code'] == 200:
                    break
                else:
                    print("OpenAI Error: ", response)
            else:
                print("Error: ", response.text)
            time.sleep(1.5)
        except requests.exceptions.ConnectionError:
            print('Connection error, retrying...')
            time.sleep(1.5)
            continue
    choices = [res['content'] for res in response['data']['response']]
    return choices


if __name__ == '__main__':
    with open('test.jsonl', 'w', encoding='utf-8') as writer:
        pass
