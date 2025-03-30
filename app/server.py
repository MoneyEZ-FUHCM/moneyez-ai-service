from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.exception.exception import register_exception_handlers
from .add_langgraph_route import add_langgraph_route

from .knowledge.routes import router as knowledge_router
from .langgraph.agent import assistant_ui_graph

print("\n[SERVER] Initializing FastAPI application")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("[SERVER] Added CORS middleware")

# Register custom exception handlers
register_exception_handlers(app)
print("[SERVER] Registered custom exception handlers")

print("[SERVER] Adding LangGraph routes")
add_langgraph_route(app, assistant_ui_graph, "/api")


# Include routers
app.include_router(knowledge_router, prefix="/api/knowledge", tags=["knowledge"])

print("[SERVER] Added knowledge router")

if __name__ == "__main__":
    import uvicorn

    print("[SERVER] Starting Uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
