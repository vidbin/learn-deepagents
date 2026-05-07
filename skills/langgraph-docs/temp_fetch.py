import urllib.request

url = "https://docs.langchain.com/llms.txt"
try:
    with urllib.request.urlopen(url, timeout=30) as response:
        content = response.read().decode('utf-8')
        print(content)
except Exception as e:
    print(f"Error: {e}")
