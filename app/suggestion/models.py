from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class QAPair(BaseModel):
    """Question and answer pair for user profile analysis."""
    question: str
    answer: str

class SuggestionRequest(BaseModel):
    """Request body for spending model suggestion."""
    data: str

class SpendingModel(BaseModel):
    """Spending model data structure."""
    id: str
    name: str
    description: str

class SuggestionResponse(BaseModel):
    """Response for spending model suggestion."""
    recommendedModel: SpendingModel
    alternativeModels: List[SpendingModel]
    reasoning: str