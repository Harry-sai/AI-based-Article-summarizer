from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from urllib.parse import urlparse
from summarizer import ArticleSummarizer

app = FastAPI(title="Article Summarization API")

# ---------- Request / Response ----------
class SummarizeRequest(BaseModel):
    url: str
    title: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str):
        parsed = urlparse(v)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("Invalid URL")
        return v

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str):
        if len(v.strip()) < 5:
            raise ValueError("Title must be at least 5 characters")
        return v.strip()


class SummarizeResponse(BaseModel):
    summary: str


# ---------- Endpoint ----------
@app.post("/summarize", response_model=SummarizeResponse)
def summarize_article(req: SummarizeRequest):
    try:
        summarizer = ArticleSummarizer(req.title, req.url)
        summary = summarizer.stream_article()
        return {"summary": summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
