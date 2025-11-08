from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse

from langchain_openai import ChatOpenAI
# from langchain.schema import AIMessage, HumanMessage
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

app = FastAPI()

llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)

class ChatRequest(BaseModel):
    messages: List[str]

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    system_prompt = """
    You are an expert software architect and design assistant. Your task is to analyze high-level user requirements and automatically design appropriate software solutions.

    Using advanced software engineering principles. Do not answer any other questions not related to software architecture and design. You should:

    - Interpret the given user requirements clearly.
    - Identify the suitable software architecture style (e.g., microservices, layered, event-driven, serverless, etc.) and justify your choice.
    - Propose a solution architecture diagram or structured textual design including components, data flow, APIs, and databases.
    - Recommend appropriate technologies, frameworks, and design patterns for each component.
    - Explain the reasoning behind each design decision, including scalability, maintainability, performance, and security aspects.
    - Optionally, generate starter code snippets or configuration examples (in Python, Java, or TypeScript).
    """

    conversation = [AIMessage(content=system_prompt)]
    for msg in request.messages:
        conversation.append(HumanMessage(content=msg))

    def generate():
        for chunk in llm.stream(conversation):
            if chunk.content:
                yield chunk.content

    return StreamingResponse(generate(), media_type="text/plain")

