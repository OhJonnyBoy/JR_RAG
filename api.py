from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from threading import Lock, Thread
from uuid import uuid4

from jr_analysis import run_rag_analysis, run_tailored_cv_generation
from google_docs_export import export_text_to_google_doc


class AnalyzeRequest(BaseModel):
    job_description: str


class TailorCvRequest(BaseModel):
    job_description: str
    target_word_budget: int = 1000
    max_bullets_per_role: int = 4
    export_to_google_doc: bool = False
    google_doc_title: str = "Tailored CV Draft"


class AnalyzeResponse(BaseModel):
    result: str
    chunks: list[str]


class TailorCvResponse(BaseModel):
    tailored_cv: str
    selected_chunks: list[str]
    selected_word_count: int
    google_doc_url: str | None = None


class AnalyzeAndTailorResponse(BaseModel):
    analysis_result: str
    analysis_chunks: list[str]
    tailored_cv: str
    selected_chunks: list[str]
    selected_word_count: int
    google_doc_url: str | None = None


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


@app.post("/tailor-cv", response_model=TailorCvResponse)
def tailor_cv(request: TailorCvRequest) -> TailorCvResponse:
    job_description = request.job_description.strip()
    if not job_description:
        raise HTTPException(status_code=400, detail="job_description cannot be empty")
    if request.target_word_budget <= 0:
        raise HTTPException(status_code=400, detail="target_word_budget must be > 0")
    if request.max_bullets_per_role <= 0:
        raise HTTPException(status_code=400, detail="max_bullets_per_role must be > 0")

    try:
        tailored_cv, selected_chunks, selected_word_count = run_tailored_cv_generation(
            job_description=job_description,
            target_word_budget=request.target_word_budget,
            max_bullets_per_role=request.max_bullets_per_role,
            return_chunks=True,
        )
        google_doc_url = None
        if request.export_to_google_doc:
            google_doc_url = export_text_to_google_doc(
                text=tailored_cv,
                title=request.google_doc_title,
            )
        return TailorCvResponse(
            tailored_cv=tailored_cv,
            selected_chunks=selected_chunks,
            selected_word_count=selected_word_count,
            google_doc_url=google_doc_url,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/analyze-and-tailor", response_model=AnalyzeAndTailorResponse)
def analyze_and_tailor(request: TailorCvRequest) -> AnalyzeAndTailorResponse:
    job_description = request.job_description.strip()
    if not job_description:
        raise HTTPException(status_code=400, detail="job_description cannot be empty")
    if request.target_word_budget <= 0:
        raise HTTPException(status_code=400, detail="target_word_budget must be > 0")
    if request.max_bullets_per_role <= 0:
        raise HTTPException(status_code=400, detail="max_bullets_per_role must be > 0")

    try:
        analysis_result, analysis_chunks = run_rag_analysis(job_description, return_chunks=True)
        tailored_cv, selected_chunks, selected_word_count = run_tailored_cv_generation(
            job_description=job_description,
            target_word_budget=request.target_word_budget,
            max_bullets_per_role=request.max_bullets_per_role,
            return_chunks=True,
        )
        google_doc_url = None
        if request.export_to_google_doc:
            google_doc_url = export_text_to_google_doc(
                text=tailored_cv,
                title=request.google_doc_title,
            )
        return AnalyzeAndTailorResponse(
            analysis_result=analysis_result,
            analysis_chunks=analysis_chunks,
            tailored_cv=tailored_cv,
            selected_chunks=selected_chunks,
            selected_word_count=selected_word_count,
            google_doc_url=google_doc_url,
        )
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
