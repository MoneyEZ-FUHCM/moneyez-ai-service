import re
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
import requests
import json
from langchain.prompts import PromptTemplate

def get_user_subcategories(user_id):
    """Fetch subcategories for a given user ID from the API."""
    if not user_id:
        print("[TOOLS] Warning: user_id is empty")
        # Fallback to a default user_id or return empty list
        return "No subcategories available"
    
    print(f"\n[TOOLS] get_user_subcategories called for user: {user_id}")
    headers = {"X-External-Secret": "thisIsSerectKeyPythonService"}
    url = f"https://easymoney.anttravel.online/api/v1/external-services?command=get_subcategories&query=user_id={user_id}"

    print(f"[TOOLS] Making API request to: {url}")
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        print(f"[TOOLS] API request successful, status code: {response.status_code}")
        subcategories = response.json().get("data", [])
        print(f"[TOOLS] Retrieved {len(subcategories)} subcategories")
        formatted_subcategories = []
        for sc in subcategories:
            formatted_subcategories.append(f"""Là một danh mục con nằm trong danh mục {sc.get("categoryName")}, danh mục này có tên là {sc.get("name")}, mã danh mục là {sc.get("code")}, mô tả là {sc.get("description")}""")
        return "\n".join(formatted_subcategories)
    else:
        print(f"[TOOLS] API request failed, status code: {response.status_code}")
        print(f"[TOOLS] API request failed, response: {response.data.message}")
        return None

def parse_response(response):
    """Parse the JSON response from the LLM."""
    try:
        response = re.sub(r"```", "", response)
        response = response.replace("\n", "").replace("json", "")
        parsed_response = json.loads(response)
        return parsed_response
    except ValueError:
        print("[ERROR] Failed to parse JSON response")
        return None

@tool(return_direct=False)
def user_input_expense(user_query: str):
    """Process user input for expenses and classify them."""
    from app.add_langgraph_route import userId
    print(f"[TOOLS] user_input_expense called with user_id: {userId}")

    if userId is None:
        return "User ID is not set. Please provide a valid user ID."
    
    subcategories = get_user_subcategories(userId)
    subllm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    template = PromptTemplate(
        input_variables=["subcategories", "user_query"],
        template="""
        Bạn là một trợ lý tài chính thông minh. Nhiệm vụ của bạn là phân tích chi tiêu của người dùng và phân loại vào danh mục thích hợp.
        
        Dưới đây là các danh mục chi tiêu có sẵn:
        {subcategories}
        
        Người dùng vừa nhập: "{user_query}"
        
        Hãy phân tích thông tin này và trả về kết quả dưới dạng JSON với cấu trúc sau:
        {{
            "amount": [số tiền chi tiêu, chỉ bao gồm con số],
            "subcategory_code": [mã danh mục phù hợp nhất]
        }}
        
        Chỉ trả về đúng định dạng JSON yêu cầu, không thêm bất kỳ giải thích nào khác.
        lưu ý các từ ngữ có thể chỉ tiền như k, lít, củ, xị,..... của ngôn ngữ tiếng việt đơn vị là VNĐ, nhỏ nhất là 1000 VNĐ
        Nếu không thể xác định được số tiền hoặc danh mục, hãy gán giá trị null cho trường tương ứng.
        """
    )
    prompt = template.format(
        subcategories=subcategories,
        user_query=user_query
    )
    response = subllm.invoke(prompt)
    print(f"[TOOLS] LLM response: {response.content}")
    parsed_response = parse_response(response.content)

    url = "https://easymoney.anttravel.online/api/v1/external-services"
    headers = {"X-External-Secret": "thisIsSerectKeyPythonService"}
    body = {
        "command": "create_transaction",
        "data": {
            "UserId": userId,
            "Amount": parsed_response.get("amount"),
            "SubcategoryCode": parsed_response.get("subcategory_code"),
            "Description": user_query
        }
    }
    response = requests.post(url, headers=headers, json=body)
    print(f"body: {body}")
    
    if response.status_code == 200:
        print(f"[TOOLS] API request successful, status code: {response.status_code}")
    else:
        print(f"[TOOLS] API request failed, status code: {response.status_code}")
        print(f"[TOOLS] API request failed, response: {response.content}")
    print(f"[TOOLS] API request body: {body}")

    return parsed_response

tools = [user_input_expense]
