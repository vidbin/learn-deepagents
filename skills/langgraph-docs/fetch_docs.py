import urllib.request

url = "https://docs.langchain.com/llms.txt"
try:
    with urllib.request.urlopen(url, timeout=30) as response:
        content = response.read().decode('utf-8')
        with open('/skills/langgraph-docs/docs_output.txt', 'w', encoding='utf-8') as f:
            f.write(content)
        print("Success! Content saved to /skills/langgraph-docs/docs_output.txt")
except Exception as e:
    print(f"Error: {e}")
