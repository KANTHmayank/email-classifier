from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List
from models import EmailClassifier
from utils import mask_pii

router = APIRouter()

# ✅ Initialize the classifier once with a valid HF model
email_classifier = EmailClassifier(model_name="distilbert-base-uncased-finetuned-sst-2-english")

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

        # ✅ Call the classify method properly
        category = email_classifier.classify(masked_email)

        return ClassifyResponse(
            input_email_body=body.input_email_body,
            list_of_masked_entities=entities,
            masked_email=masked_email,
            category_of_the_email=category
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
