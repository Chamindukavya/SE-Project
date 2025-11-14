from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict
from dotenv import load_dotenv
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

app = FastAPI()

# LLM for streaming chat/textual interaction
llm_stream = ChatOpenAI(model="gpt-4o-mini", streaming=True)

# LLM instance for structured output (non-streaming)
llm_structured = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class ChatRequest(BaseModel):
    messages: List[str]

class Component(BaseModel):
    id: str = Field(..., description="Unique identifier used for edges")
    name: str
    type: str = Field(..., description="Category e.g. service, database, gateway")
    description: str
    tech_stack: List[str]
    depends_on: List[str] = Field(default_factory=list, description="List of component ids this component depends on")

class ArchitectureDesign(BaseModel):
    architecture_style: str
    reasoning: str
    components: List[Component]

class DesignRequest(BaseModel):
    requirements: str = Field(..., description="High-level user requirements text")

def build_system_prompt() -> str:
    return (
        "You are an expert software architect and design assistant. Given high-level user requirements, "
        "produce a structured JSON architecture design following this schema: "
        "ArchitectureDesign -> architecture_style (string), reasoning (string), components (array of Component). "
        "Each Component: id (kebab-case unique), name (readable), type (service|database|cache|frontend|gateway|queue|topic|function|storage|other), description, tech_stack (array), depends_on (array of ids). "
        "Choose an appropriate architecture style and justify it in reasoning. Focus on scalability, maintainability, performance, security. "
        "Only output valid JSON, no markdown, no commentary outside JSON."
    )

def topological_levels(components: List[Component]) -> Dict[str, int]:
    # Simple DAG level computation; if cycles, they all get max depth encountered.
    deps = {c.id: set(c.depends_on) for c in components}
    level: Dict[str, int] = {c.id: 0 for c in components}
    changed = True
    # Relaxation approach; depth <= number of components iterations
    for _ in range(len(components)):
        if not changed:
            break
        changed = False
        for cid, d in deps.items():
            if d:
                new_level = max(level.get(x, 0) for x in d) + 1
                if new_level > level[cid]:
                    level[cid] = new_level
                    changed = True
    return level

def to_react_flow(design: ArchitectureDesign) -> Dict[str, List[Dict]]:
    levels = topological_levels(design.components)
    # Group by level for vertical placement
    grouped: Dict[int, List[Component]] = {}
    for c in design.components:
        grouped.setdefault(levels[c.id], []).append(c)

    nodes = []
    level_x_gap = 300
    item_y_gap = 140
    for lvl, comps in grouped.items():
        for idx, comp in enumerate(comps):
            nodes.append({
                "id": comp.id,
                "type": "default",
                "position": {"x": lvl * level_x_gap, "y": idx * item_y_gap},
                "data": {
                    "label": f"{comp.name}\n({comp.type})",
                    "description": comp.description,
                    "tech_stack": comp.tech_stack,
                },
            })

    edges = []
    for comp in design.components:
        for dep in comp.depends_on:
            if any(d.id == dep for d in design.components):  # only include valid dependencies
                edge_id = f"{dep}__{comp.id}"
                edges.append({
                    "id": edge_id,
                    "source": dep,
                    "target": comp.id,
                    "animated": True,
                })
    return {"nodes": nodes, "edges": edges}

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
        for chunk in llm_stream.stream(conversation):
            if chunk.content:
                yield chunk.content

    return StreamingResponse(generate(), media_type="text/plain")

@app.post("/architecture")
async def design_endpoint(req: DesignRequest):
    system_prompt = build_system_prompt()
    messages = [
        AIMessage(content=system_prompt),
        HumanMessage(content=req.requirements)
    ]
    # Structured output via JSON mode (LangChain will coerce response to dict matching schema)
    structured_llm = llm_structured.with_structured_output(ArchitectureDesign)
    design: ArchitectureDesign = structured_llm.invoke(messages)  # type: ignore

    react_flow = to_react_flow(design)
    response = {
        "architecture_style": design.architecture_style,
        "reasoning": design.reasoning,
        "components": [c.model_dump() for c in design.components],
        "react_flow": react_flow,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    return JSONResponse(content=response)

