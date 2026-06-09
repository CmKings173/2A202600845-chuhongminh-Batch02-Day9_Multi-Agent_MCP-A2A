"""
Lab_Assignment/test_agent.py — Test script cho Supervisor-Workers Agent
=======================================================================

Kiểm thử 3 loại câu hỏi:
  1. Câu hỏi thuần về luật ma tuý → legal_worker
  2. Câu hỏi về nghệ sĩ            → news_worker
  3. Câu hỏi kết hợp               → cả hai workers
"""

from __future__ import annotations

import os
import sys
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import run_agent

TEST_QUESTIONS = [
    {
        "id": 1,
        "label": "Câu hỏi pháp luật",
        "question": "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo Điều 249 Bộ luật Hình sự là gì?",
        "expected_worker": "legal_worker",
    },
    {
        "id": 2,
        "label": "Câu hỏi nghệ sĩ",
        "question": "Nghệ sĩ Việt Nam nào đã bị bắt liên quan đến ma tuý gần đây?",
        "expected_worker": "news_worker",
    },
    {
        "id": 3,
        "label": "Câu hỏi kết hợp",
        "question": "Các nghệ sĩ bị bắt vì ma tuý đã vi phạm điều khoản nào của luật hình sự?",
        "expected_worker": "both workers",
    },
]


def run_test(test: dict) -> None:
    print(f"\n{'='*70}")
    print(f"[TEST {test['id']}] {test['label']}")
    print(f"Expected Workers: {test['expected_worker']}")
    print(f"Câu hỏi: {test['question']}")
    print("-" * 70)

    start = time.time()
    result = run_agent(test["question"])
    elapsed = time.time() - start

    # Kiểm tra workers đã được gọi
    ctx = result.get("context", "")
    workers_called = []
    if "## Legal Search Results" in ctx:
        workers_called.append("legal_worker")
    if "## News Search Results" in ctx:
        workers_called.append("news_worker")
    if "## PageIndex Results" in ctx:
        workers_called.append("pageindex_worker")

    print(f"✅ Workers gọi: {workers_called or ['(none)']}")
    print(f"✅ Iterations:  {result.get('iterations', 0)}")
    print(f"✅ Thời gian:   {elapsed:.1f}s")
    print(f"\n📋 KẾT QUẢ:\n{result['final_answer']}")


def main():
    print("🚀 Bắt đầu kiểm thử Supervisor-Workers Agent")
    print(f"Số câu hỏi: {len(TEST_QUESTIONS)}\n")

    for test in TEST_QUESTIONS:
        run_test(test)

    print(f"\n{'='*70}")
    print("✅ Hoàn tất kiểm thử!")


if __name__ == "__main__":
    main()
