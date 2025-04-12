import json
from fastapi import APIRouter, status

from app.exception.exception import CustomHTTPException
from app.models import BaseResponse

from .models import SuggestionRequest, SuggestionResponse, QAPair
from .service import generate_suggestion

router = APIRouter()

@router.post("/api/suggestion", response_model=BaseResponse[SuggestionResponse])
async def suggest_spending_model(request: SuggestionRequest):
    """
    Analyze Q&A pairs and suggest appropriate spending models for the user.
    Fetches spending models from the external API and uses AI to determine the best match.
    
    The request should contain a 'data' field with a JSON string.
    Example: {"data": "[{\"question\":\"Thu nhập?\",\"answer\":\"15 triệu\"}]"}
    """
    try:
        print(f"\n[SUGGESTION API] Received request with data: {request.data[:100]}...")
        
        # Parse JSON string từ trường data
        try:
            parsed_data = json.loads(request.data)
            qa_pairs = []
            
            # Chuyển đổi dữ liệu đã parse thành danh sách các đối tượng QAPair
            for item in parsed_data:
                if isinstance(item, dict) and "question" in item and "answer" in item:
                    qa_pairs.append(QAPair(
                        question=item["question"], 
                        answer=item["answer"]
                    ))
            
            print(f"[SUGGESTION API] Successfully parsed {len(qa_pairs)} Q&A pairs")
            
            if not qa_pairs:
                raise ValueError("No valid Q&A pairs found in the data")
                
        except json.JSONDecodeError as e:
            print(f"[SUGGESTION API] JSON parsing error: {str(e)}")
            return BaseResponse(
                status=status.HTTP_400_BAD_REQUEST,
                error_code="INVALID_JSON",
                message=f"Invalid JSON format in 'data' field: {str(e)}"
            )
        
        # Gọi service để xử lý gợi ý mô hình chi tiêu
        result = await generate_suggestion(qa_pairs)
        
        print(f"[SUGGESTION API] Returning recommendation: {result.recommendedModel.name}")
        
        return BaseResponse(
            status=status.HTTP_200_OK,
            message="Spending model suggestion generated successfully",
            data=result
        )
            
    except Exception as e:
        print(f"[SUGGESTION API] Unexpected error: {str(e)}")
        return BaseResponse(
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="SUGGESTION_ERROR",
            message=f"Error generating spending model suggestion: {str(e)}"
        )