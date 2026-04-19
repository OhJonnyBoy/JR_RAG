from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from threading import Lock, Thread
from uuid import uuid4

from jr_analysis import run_rag_analysis


class AnalyzeRequest(BaseModel):
    job_description: str


class AnalyzeResponse(BaseModel):
    result: str
    chunks: list[str]


class AnalyzeStartResponse(BaseModel):
    job_id: str


class AnalyzeStatusResponse(BaseModel):
    job_id: str
    status: str
    is_done: bool
    result: str | None = None
    chunks: list[str] | None = None
    error: str | None = None


app = FastAPI(title="JR RAG API")
jobs: dict[str, dict] = {}
jobs_lock = Lock()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    job_description = request.job_description.strip()
    if not job_description:
        raise HTTPException(status_code=400, detail="job_description cannot be empty")

    try:
        result, chunks = run_rag_analysis(job_description, return_chunks=True)
        return AnalyzeResponse(result=result, chunks=chunks)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _run_analysis_job(job_id: str, job_description: str) -> None:
    def update_status(message: str) -> None:
        with jobs_lock:
            jobs[job_id]["status"] = message

    try:
        result, chunks = run_rag_analysis(
            job_description,
            return_chunks=True,
            status_callback=update_status,
        )
        with jobs_lock:
            jobs[job_id]["result"] = result
            jobs[job_id]["chunks"] = chunks
            jobs[job_id]["status"] = "Completed"
            jobs[job_id]["is_done"] = True
    except Exception as exc:
        with jobs_lock:
            jobs[job_id]["status"] = "Failed"
            jobs[job_id]["error"] = str(exc)
            jobs[job_id]["is_done"] = True


@app.post("/analyze/start", response_model=AnalyzeStartResponse)
def analyze_start(request: AnalyzeRequest) -> AnalyzeStartResponse:
    job_description = request.job_description.strip()
    if not job_description:
        raise HTTPException(status_code=400, detail="job_description cannot be empty")

    job_id = str(uuid4())
    with jobs_lock:
        jobs[job_id] = {
            "status": "Queued",
            "is_done": False,
            "result": None,
            "chunks": None,
            "error": None,
        }

    thread = Thread(target=_run_analysis_job, args=(job_id, job_description), daemon=True)
    thread.start()
    return AnalyzeStartResponse(job_id=job_id)


@app.get("/analyze/status/{job_id}", response_model=AnalyzeStatusResponse)
def analyze_status(job_id: str) -> AnalyzeStatusResponse:
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job_id not found")

        return AnalyzeStatusResponse(
            job_id=job_id,
            status=job["status"],
            is_done=job["is_done"],
            result=job["result"],
            chunks=job["chunks"],
            error=job["error"],
        )
