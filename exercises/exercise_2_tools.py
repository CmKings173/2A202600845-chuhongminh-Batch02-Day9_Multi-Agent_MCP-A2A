"""Bài Tập 2: Thêm Tools và Knowledge Base

Hoàn thành các TODO để thêm tool và knowledge base entry mới.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from common.llm import get_llm

# Knowledge base
LEGAL_KNOWLEDGE = [
    {
        "id": "ucc_breach",
        "keywords": ["breach", "contract", "remedies", "damages", "ucc"],
        "text": (
            "Under the Uniform Commercial Code (UCC) Article 2, remedies for breach of contract "
            "include: (1) expectation damages; (2) consequential damages; (3) specific performance; "
            "(4) cover damages. Statute of limitations is typically 4 years (UCC § 2-725)."
        ),
    },
    # TODO: Thêm entry về luật lao động Việt Nam
    # Gợi ý: id="labor_law", keywords=["lao động", "sa thải", ...], text="..."
    {
        "id": "labor_law",
        "keywords": ["lao động", "sa thải", "hợp đồng lao động", "chấm dứt"],
        "text": (
            "Theo Bộ luật Lao động Việt Nam 2019, người sử dụng lao động chỉ được đơn phương chấm dứt "
            "hợp đồng lao động trong các trường hợp được pháp luật quy định (như người lao động thường xuyên không hoàn thành công việc, "
            "bị ốm đau tai nạn kéo dài, hoặc do thiên tai dịch bệnh). Người sử dụng lao động phải báo trước ít nhất 45 ngày đối với hợp đồng không xác định thời hạn, "
            "30 ngày đối với hợp đồng xác định thời hạn từ 12-36 tháng, và 3 ngày làm việc đối với hợp đồng dưới 12 tháng."
        ),
    }
]


@tool
def search_legal_knowledge(query: str) -> str:
    """Tìm kiếm trong knowledge base pháp lý."""
    query_lower = query.lower()
    for entry in LEGAL_KNOWLEDGE:
        if any(kw in query_lower for kw in entry["keywords"]):
            return f"[{entry['id']}] {entry['text']}"
    return "Không tìm thấy thông tin liên quan."


# TODO: Tạo tool check_statute_of_limitations
# Gợi ý: nhận case_type (str), trả về thời hiệu khởi kiện
@tool
def check_statute_of_limitations(case_type: str) -> str:
    """Kiểm tra thời hiệu khởi kiện cho các loại tranh chấp dân sự, lao động hoặc thương mại.

    Args:
        case_type: Loại tranh chấp hoặc vụ việc (ví dụ: 'hợp đồng', 'dân sự', 'lao động', 'thương mại').
    """
    case_type_lower = case_type.lower()
    if "hợp đồng" in case_type_lower or "dân sự" in case_type_lower:
        return "Thời hiệu khởi kiện để yêu cầu Tòa án giải quyết tranh chấp hợp đồng là 03 năm, kể từ ngày người có quyền yêu cầu biết hoặc phải biết quyền và lợi ích hợp pháp của mình bị xâm phạm (Điều 429 Bộ luật Dân sự 2015)."
    elif "lao động" in case_type_lower:
        return "Thời hiệu khởi kiện về lao động là 01 năm kể từ ngày phát hiện hành vi xâm phạm quyền và lợi ích hợp pháp (Điều 190 Bộ luật Lao động 2019)."
    elif "thương mại" in case_type_lower or "kinh doanh" in case_type_lower:
        return "Thời hiệu khởi kiện đối với tranh chấp thương mại là 02 năm kể từ thời điểm quyền và lợi ích hợp pháp bị xâm phạm (Điều 319 Luật Thương mại 2005)."
    else:
        return "Thời hiệu khởi kiện dân sự chung là 03 năm kể từ ngày người có quyền yêu cầu biết hoặc phải biết quyền và lợi ích hợp pháp bị xâm phạm (Điều 184 Bộ luật Dân sự 2015)."


async def main():
    load_dotenv()
    llm = get_llm()
    
    # TODO: Thêm tool mới vào danh sách
    tools = [search_legal_knowledge, check_statute_of_limitations]  # Thêm check_statute_of_limitations vào đây
    llm_with_tools = llm.bind_tools(tools)
    
    question = "Thời hiệu khởi kiện vụ vi phạm hợp đồng là bao lâu?"
    
    messages = [
        SystemMessage(content="Bạn là chuyên gia pháp lý. Sử dụng tools để tra cứu thông tin."),
        HumanMessage(content=question),
    ]
    
    print(f"Câu hỏi: {question}\n")
    
    # First LLM call - decide which tools to use
    response = await llm_with_tools.ainvoke(messages)
    messages.append(response)
    
    # Execute tools if requested
    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"🔧 Gọi tool: {tool_call['name']}")
            tool_result = None
            
            if tool_call["name"] == "search_legal_knowledge":
                tool_result = search_legal_knowledge.invoke(tool_call["args"])
            # TODO: Thêm xử lý cho check_statute_of_limitations
            elif tool_call["name"] == "check_statute_of_limitations":
                tool_result = check_statute_of_limitations.invoke(tool_call["args"])
            
            if tool_result:
                messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))
        
        # Second LLM call - synthesize final answer
        final_response = await llm_with_tools.ainvoke(messages)
        print(f"\n✅ Kết quả:\n{final_response.content}")
    else:
        print(f"\n✅ Kết quả:\n{response.content}")


if __name__ == "__main__":
    asyncio.run(main())
