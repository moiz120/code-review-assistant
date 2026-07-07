"""
Dataset Preparation Script for Code Review Fine-tuning.

This script helps you:
1. Scrape PR reviews from GitHub repositories
2. Clean and format the data
3. Create train/test splits
4. Export in the format needed for fine-tuning

Usage:
    python prepare_dataset.py --repo owner/repo --output ./data/reviews.json
"""
import os
import json
import argparse
from datetime import datetime
from typing import List, Dict
import requests


def fetch_github_reviews(repo: str, token: str, max_prs: int = 100) -> List[Dict]:
    """
    Fetch pull request reviews from a GitHub repository.

    Args:
        repo: Repository in format "owner/repo"
        token: GitHub personal access token
        max_prs: Maximum number of PRs to fetch

    Returns:
        List of review data dictionaries
    """
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    reviews = []
    page = 1

    print(f"Fetching PRs from {repo}...")

    while len(reviews) < max_prs:
        # Fetch PRs
        url = f"https://api.github.com/repos/{repo}/pulls"
        params = {
            "state": "closed",
            "per_page": 30,
            "page": page,
            "sort": "updated",
            "direction": "desc"
        }

        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            print(f"Error fetching PRs: {response.status_code}")
            break

        prs = response.json()

        if not prs:
            break

        for pr in prs:
            pr_number = pr["number"]

            # Fetch PR files (diffs)
            files_url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/files"
            files_response = requests.get(files_url, headers=headers)

            if files_response.status_code != 200:
                continue

            files = files_response.json()

            # Fetch PR reviews
            reviews_url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/reviews"
            reviews_response = requests.get(reviews_url, headers=headers)

            if reviews_response.status_code != 200:
                continue

            pr_reviews = reviews_response.json()

            # Process each file with its reviews
            for file_data in files:
                if file_data.get("status") == "removed":
                    continue

                # Find reviews for this file
                file_reviews = [
                    r for r in pr_reviews 
                    if r.get("path") == file_data.get("filename")
                ]

                for review in file_reviews:
                    if review.get("body"):
                        reviews.append({
                            "repo": repo,
                            "pr_number": pr_number,
                            "file": file_data.get("filename"),
                            "patch": file_data.get("patch", ""),
                            "review_body": review["body"],
                            "state": review.get("state"),
                            "created_at": review.get("created_at"),
                            "author": review.get("user", {}).get("login")
                        })

        print(f"  Fetched page {page}, total reviews: {len(reviews)}")
        page += 1

        if len(prs) < 30:
            break

    return reviews


def clean_review_data(reviews: List[Dict]) -> List[Dict]:
    """
    Clean and filter review data.

    Removes:
    - Empty or very short reviews
    - Bot-generated reviews
    - Reviews without code context
    """
    cleaned = []

    bot_names = ["github-actions", "dependabot", "renovate", "codecov"]

    for review in reviews:
        # Skip bot reviews
        if review.get("author") in bot_names:
            continue

        # Skip empty or very short reviews
        body = review.get("review_body", "").strip()
        if len(body) < 20:
            continue

        # Skip reviews without patches
        if not review.get("patch"):
            continue

        # Clean the review text
        body = body.replace("\r\n", "\n").replace("\r", "\n")

        cleaned.append({
            **review,
            "review_body": body
        })

    return cleaned


def format_for_training(reviews: List[Dict]) -> List[Dict]:
    """
    Format reviews for fine-tuning.

    Creates instruction-following format:
    {
        "instruction": "Review the following code:",
        "input": "<code>",
        "output": "<review>"
    }
    """
    formatted = []

    for review in reviews:
        # Extract code from patch
        patch = review.get("patch", "")
        code_lines = []

        for line in patch.split("\n"):
            if line.startswith("+") and not line.startswith("+++"):
                code_lines.append(line[1:])

        code = "\n".join(code_lines)

        if len(code) < 10:  # Skip if too little code
            continue

        formatted.append({
            "instruction": f"Review the following {review['file'].split('.')[-1]} code:",
            "input": code[:2000],  # Limit code length
            "output": review["review_body"][:1000],  # Limit review length
            "metadata": {
                "repo": review["repo"],
                "pr": review["pr_number"],
                "file": review["file"]
            }
        })

    return formatted


def main():
    parser = argparse.ArgumentParser(description="Prepare code review dataset")
    parser.add_argument("--repo", type=str, required=True, help="GitHub repo (owner/repo)")
    parser.add_argument("--token", type=str, default=os.getenv("GITHUB_TOKEN"), help="GitHub token")
    parser.add_argument("--output", type=str, default="./data/reviews.json", help="Output file")
    parser.add_argument("--max-prs", type=int, default=100, help="Max PRs to fetch")

    args = parser.parse_args()

    if not args.token:
        print("Error: GitHub token required. Set GITHUB_TOKEN env var or use --token")
        return

    # Fetch reviews
    raw_reviews = fetch_github_reviews(args.repo, args.token, args.max_prs)
    print(f"\nFetched {len(raw_reviews)} raw reviews")

    # Clean
    cleaned = clean_review_data(raw_reviews)
    print(f"Cleaned: {len(cleaned)} reviews")

    # Format
    formatted = format_for_training(cleaned)
    print(f"Formatted for training: {len(formatted)} examples")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(formatted, f, indent=2)

    print(f"\nSaved to: {args.output}")
    print(f"\nDataset statistics:")
    print(f"  Total examples: {len(formatted)}")
    print(f"  Avg code length: {sum(len(r['input']) for r in formatted) / len(formatted):.0f} chars")
    print(f"  Avg review length: {sum(len(r['output']) for r in formatted) / len(formatted):.0f} chars")


if __name__ == "__main__":
    main()
