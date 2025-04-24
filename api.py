from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List
from models import classify_email
from utils import mask_pii

router = APIRouter()

class ClassifyRequest(BaseModel):
    input_email_body: str = Field(..., example="Hi, I have a billing question...")

class Entity(BaseModel):
    position: List[int]
    classification: str
    entity: str

class ClassifyResponse(BaseModel):
    input_email_body: str
    list_of_masked_entities: List[Entity]
    masked_email: str
    category_of_the_email: str

@router.post("/classify", response_model=ClassifyResponse)
async def classify_endpoint(body: ClassifyRequest):
    try:
        masked_email, entities = mask_pii(body.input_email_body)
        category = classify_email(masked_email)
        return ClassifyResponse(
            input_email_body=body.input_email_body,
            list_of_masked_entities=entities,
            masked_email=masked_email,
            category_of_the_email=category
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
