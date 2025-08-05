from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from utils import load_pages, add_page, delete_page, rebuild_index, load_query_engine
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Agente WIki", version="1.0.0")

origins = os.getenv("ALLOWED_ORIGINS", "").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # permite frontend
    allow_credentials=True,
    allow_methods=["*"],     # permite todos los métodos, incluido OPTIONS
    allow_headers=["*"],
)

class PageRequest(BaseModel):
    page: str

class QuestionRequest(BaseModel):
    question: str

@app.get("/pages")
def get_pages():
    return {"pages": load_pages()}

@app.post("/pages")
def add_new_page(req: PageRequest):
    success = add_page(req.page)
    if not success:
        raise HTTPException(status_code=400, detail="Page already exists.")
    return {"message": f"Page '{req.page}' added and index updated."}

@app.delete("/pages")
def remove_page(req: PageRequest):
    success = delete_page(req.page)
    if not success:
        raise HTTPException(status_code=404, detail="Page not found.")
    return {"message": f"Page '{req.page}' removed and index updated."}

@app.post("/rebuild-index")
def rebuild():
    rebuild_index()
    return {"message": "Index rebuilt from existing pages."}

@app.post("/ask")
def ask_question(req: QuestionRequest):
    query_engine = load_query_engine()
    response = query_engine.query(req.question)
    return {
        "response": response.response,
        "sources": [src.node.get_content() for src in response.source_nodes]
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)