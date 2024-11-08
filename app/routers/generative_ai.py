from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from groq import Groq
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

router = APIRouter()

SYSTEM_PROMPT = """Eres un experto en atención al cliente con años de experiencia. 
Proporciona respuestas profesionales, empáticas y orientadas a soluciones.
Mantén un tono amable y cercano, pero siempre profesional."""

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7

class PromptResponse(BaseModel):
    response: str

@router.post("/generative-ai/generate", response_model=PromptResponse)
async def generate_response(request: PromptRequest):
    try:
        client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        
        completion = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.prompt}
            ],
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        return PromptResponse(response=completion.choices[0].message.content)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )