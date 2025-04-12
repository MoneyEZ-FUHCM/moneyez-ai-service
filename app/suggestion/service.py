import json
import requests
import re
from typing import List, Dict, Any, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

from .models import QAPair, SpendingModel, SuggestionResponse

# URL của API mô hình chi tiêu
SPENDING_MODELS_API_URL = "https://easymoney.anttravel.online/api/v1/external-services?command=get_speding_models"

def parse_json_response(response_content: str) -> Dict:
    """Parse the JSON response from the LLM."""
    try:
        # Xóa markdown code blocks nếu có
        response = re.sub(r"```json", "", response_content)
        response = re.sub(r"```", "", response)
        
        # Loại bỏ khoảng trắng và xuống dòng không cần thiết
        response = response.strip()
        
        # Parse JSON
        parsed_response = json.loads(response)
        return parsed_response
    except ValueError as e:
        print(f"[SUGGESTION] Failed to parse JSON response: {str(e)}")
        # Thử tìm đoạn JSON trong văn bản
        try:
            json_match = re.search(r'\{.*"recommended_model_id".*\}', response_content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except:
            pass
        return None

async def get_spending_models() -> List[Dict[str, Any]]:
    """Lấy danh sách các mô hình chi tiêu từ API."""
    try:
        headers = {"X-External-Secret": "thisIsSerectKeyPythonService"}
        models_response = requests.get(SPENDING_MODELS_API_URL, headers=headers)
        models_response.raise_for_status()  # Raise exception cho HTTP errors
        
        # Parse API response
        api_response = models_response.json()
        if "data" in api_response and isinstance(api_response["data"], list):
            spending_models = api_response["data"]
            print(f"[SUGGESTION] Successfully fetched {len(spending_models)} spending models")
            return spending_models
        else:
            print(f"[SUGGESTION] Unexpected API response format: {api_response}")
            raise ValueError("Invalid API response format - missing 'data' field")
            
    except Exception as e:
        print(f"[SUGGESTION] Error fetching spending models: {str(e)}")
        raise e

async def analyze_user_profile(qa_pairs: List[QAPair], spending_models: List[Dict[str, Any]]) -> Dict:
    """Phân tích thông tin người dùng và gợi ý mô hình chi tiêu phù hợp."""
    # Chuẩn bị thông tin người dùng từ các cặp Q&A
    user_profile = ""
    for idx, qa in enumerate(qa_pairs):
        user_profile += f"Q{idx+1}: {qa.question}\n"
        user_profile += f"A{idx+1}: {qa.answer}\n\n"
    
    # Sử dụng LLM để phân tích
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    template = PromptTemplate(
        input_variables=["user_profile", "spending_models"],
        template="""
        Bạn là một trợ lý tài chính thông minh. Nhiệm vụ của bạn là phân tích thông tin từ câu trả lời 
        và gợi ý mô hình chi tiêu phù hợp nhất.
        
        Dựa trên thông tin tôi đã trả lời như sau:
        {user_profile}
        
        Và các mô hình chi tiêu có sẵn:
        {spending_models}
        
        Hãy phân tích và xác định mô hình chi tiêu nào phù hợp nhất với tôi này.
        Xem xét mức thu nhập, mục tiêu tài chính, thói quen chi tiêu và tình hình tài chính tổng thể của tôi.
        Cũng đề xuất một số mô hình thay thế có thể phù hợp.
        
        Trả về kết quả dưới dạng JSON với cấu trúc sau:
        {{
            "recommended_model_name": "tên của mô hình phù hợp nhất",
            "alternative_model_names": ["name1", "name2"],
            "reasoning": "giải thích chi tiết tại sao mô hình này được đề xuất"
        }}
        
        Chỉ trả về đúng định dạng JSON yêu cầu, không thêm bất kỳ giải thích nào khác.
        """
    )
    
    prompt = template.format(
        user_profile=user_profile,
        spending_models=json.dumps(spending_models, ensure_ascii=False)
    )
    
    print(f"[SUGGESTION] Generating recommendation using AI model")
    response = await llm.ainvoke(prompt)
    print(f"[SUGGESTION] LLM response: {response.content}")
    
    # Parse LLM response
    if hasattr(response, 'content') and isinstance(response.content, str):
        return parse_json_response(response.content)
    return None

async def generate_suggestion(qa_pairs: List[QAPair]) -> SuggestionResponse:
    """Tạo gợi ý mô hình chi tiêu dựa trên thông tin người dùng."""
    try:
        # Lấy danh sách mô hình chi tiêu
        spending_models = await get_spending_models()
        
        # Phân tích thông tin người dùng
        analysis_result = await analyze_user_profile(qa_pairs, spending_models)
        
        if not analysis_result:
            raise ValueError("Could not analyze user profile")
        
        # Tìm mô hình được đề xuất và các lựa chọn thay thế
        recommended_model = None
        alternative_models = []
        
        for model in spending_models:
            if model["name"] == analysis_result.get("recommended_model_name"):
                recommended_model = SpendingModel(
                    id=model["id"],
                    name=model["name"],
                    description=model["description"],
                )
            elif model["name"] in analysis_result.get("alternative_model_names", []):
                alternative_models.append(SpendingModel(
                    id=model["id"],
                    name=model["name"],
                    description=model["description"],
                ))
        
        # Nếu không tìm thấy mô hình được đề xuất, sử dụng mô hình đầu tiên
        if not recommended_model and spending_models:
            first_model = spending_models[0]
            recommended_model = SpendingModel(
                id=first_model["id"],
                name=first_model["name"],
                description=first_model["description"],
            )
        
        # Tạo response
        return SuggestionResponse(
            recommended_model=recommended_model,
            alternative_models=alternative_models,
            reasoning=analysis_result.get("reasoning", "")
        )
        
    except Exception as e:
        print(f"[SUGGESTION] Error generating suggestion: {str(e)}")
        raise e