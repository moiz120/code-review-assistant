import requests

payload = {
    "pr_payload": {
        "pr_number": 1,
        "repository": "test",
        "owner": "test",
        "title": "Test PR",
        "branch": "dev",
        "base_branch": "main",
        "author": "test",
        "changes": [{
            "filename": "bad.py",
            "status": "added",
            "additions": 5,
            "new_content": 'password = "secret123"\nimport os\nos.system("rm -rf /")'
        }]
    },
    "review_style": "strict"
}

r = requests.post("http://localhost:8000/api/v1/review", json=payload)
data = r.json()

print("Score:", data["overall_score"])
print("Comments:", len(data["comments"]))
for c in data["comments"]:
    print("  [" + c["severity"].upper() + "]", c["message"][:120]) 