"""
run_eval.py
===========
בודק איכות RAG לפי 3 מדדים:
1. Keyword coverage: האם התשובה מכילה מילות מפתח צפויות?
2. Source relevance: האם ה-chunks שנשלפו מהdoc_type הנכון?
3. Pass threshold: keyword_score >= 0.5 AND source_match

הרצה:
    python tests/eval/run_eval.py
    python tests/eval/run_eval.py --verbose
"""

import json
import sys
import argparse
import requests
from pathlib import Path

API_URL = "http://localhost:8000"
GREEN = "\033[92m"
RED   = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def _call_api(question: str, session_id: str) -> dict:
    """
    קורא ל-/chat endpoint.
    תומך בשני מצבים:
    - Streaming (SSE): מרכיב את התשובה מה-tokens
    - Batch (JSON): מחזיר ישירות
    """
    response = requests.post(
        f"{API_URL}/chat",
        json={"question": question, "session_id": session_id},
        timeout=60,
        stream=True  # תמיד stream=True כדי לתמוך בשני המצבים
    )
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")

    # SSE streaming mode
    if "text/event-stream" in content_type:
        full_answer = ""
        sources = []
        for line in response.iter_lines():
            if not line:
                continue
            if isinstance(line, bytes):
                line = line.decode("utf-8")
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    if data.get("done") is False:
                        full_answer += data.get("token", "")
                    elif data.get("done") is True:
                        sources = data.get("sources", [])
                except json.JSONDecodeError:
                    pass
        return {"answer": full_answer, "sources": sources}

    # Batch JSON mode
    else:
        return response.json()


def run_eval(verbose: bool = False) -> list[dict]:
    questions_path = Path(__file__).parent / "eval_questions.json"
    with open(questions_path) as f:
        questions = json.load(f)

    print(f"\n{GREEN}=== RAG Evaluation Set ==={RESET}")
    print(f"Questions: {len(questions)}\n")

    results = []

    for q in questions:
        try:
            response = _call_api(q["question"], f"eval-{q['id']}")
        except Exception as e:
            print(f"[{q['id']}] {RED}ERROR: {e}{RESET}")
            results.append({"id": q["id"], "pass": False, "error": str(e)})
            continue

        answer = response.get("answer", "").lower()
        sources = response.get("sources", [])

        # מדד 1: keyword coverage
        found_keywords = [kw for kw in q["expected_keywords"] if kw.lower() in answer]
        keyword_score = len(found_keywords) / len(q["expected_keywords"])

        # מדד 2: source relevance (לפי doc_type בשם הקובץ)
        source_match = any(
            doc_type in " ".join(sources).lower()
            for doc_type in q["expected_doc_types"]
        )

        passed = keyword_score >= 0.5 and source_match

        result = {
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "keyword_score": round(keyword_score, 2),
            "keywords_found": found_keywords,
            "source_match": source_match,
            "sources": sources,
            "pass": passed
        }
        results.append(result)

        # output
        status = f"{GREEN}✓ PASS{RESET}" if passed else f"{RED}✗ FAIL{RESET}"
        src_icon = "✓" if source_match else "✗"
        print(
            f"[{q['id']}] {status} | "
            f"category={q['category']:<15} | "
            f"keywords={keyword_score:.0%} | "
            f"source={src_icon}"
        )

        if verbose and not passed:
            print(f"  Question: {q['question']}")
            print(f"  Expected keywords: {q['expected_keywords']}")
            print(f"  Found: {found_keywords}")
            print(f"  Sources: {sources}")
            print()

    # Summary
    passed_count = sum(1 for r in results if r.get("pass"))
    total = len(results)
    pct = passed_count / total * 100 if total > 0 else 0

    print(f"\n{'='*50}")
    print(f"Result: {passed_count}/{total} passed ({pct:.0f}%)")

    # by category
    categories = {}
    for r in results:
        cat = r.get("category", "unknown")
        categories.setdefault(cat, {"pass": 0, "total": 0})
        categories[cat]["total"] += 1
        if r.get("pass"):
            categories[cat]["pass"] += 1

    print("\nBy category:")
    for cat, stats in categories.items():
        cat_pct = stats["pass"] / stats["total"] * 100
        color = GREEN if cat_pct >= 60 else YELLOW if cat_pct >= 40 else RED
        print(f"  {cat:<20} {stats['pass']}/{stats['total']} ({color}{cat_pct:.0f}%{RESET})")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    results = run_eval(verbose=args.verbose)

    # exit code: 0 אם >60% עברו, אחרת 1
    passed = sum(1 for r in results if r.get("pass"))
    sys.exit(0 if passed / len(results) >= 0.6 else 1)
