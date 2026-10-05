import asyncio
import os
import signal
import subprocess
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil

app = FastAPI(title="Urban Fire High-Res Pipeline API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIG_PATH = "configs/default.yaml"
LOG_FILE = "pipeline.log"
pipeline_status = {"state": "idle", "pid": None}

class ConfigData(BaseModel):
    content: str

class IngestionData(BaseModel):
    region: str
    start_date: str
    end_date: str
    max_images: int

def run_pipeline_task():
    global pipeline_status
    pipeline_status["state"] = "running"
    
    with open(LOG_FILE, "w") as f:
        process = subprocess.Popen(
            ["make", "pipeline"],
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=os.getcwd(),
            start_new_session=True
        )
        pipeline_status["pid"] = process.pid
        process.wait()
        
    pipeline_status["state"] = "completed" if process.returncode == 0 else "failed"
    pipeline_status["pid"] = None

def run_custom_pipeline_task(targets):
    global pipeline_status
    pipeline_status["state"] = "running"
    
    with open(LOG_FILE, "a") as f:
        f.write(f"\n\n--- RESUMING PIPELINE: {targets} ---\n\n")
        f.flush()
        process = subprocess.Popen(
            ["make"] + targets,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=os.getcwd(),
            start_new_session=True
        )
        pipeline_status["pid"] = process.pid
        process.wait()
        
    pipeline_status["state"] = "completed" if process.returncode == 0 else "failed"
    pipeline_status["pid"] = None

@app.get("/api/config_raw")
def get_config_raw():
    if not os.path.exists(CONFIG_PATH):
        raise HTTPException(status_code=404, detail="Config not found")
    with open(CONFIG_PATH, "r") as f:
        return {"content": f.read()}

@app.post("/api/config_raw")
def update_config_raw(data: ConfigData):
    with open(CONFIG_PATH, "w") as f:
        f.write(data.content)
    return {"status": "success"}

@app.post("/api/update_ingestion")
def update_ingestion(data: IngestionData):
    with open(CONFIG_PATH, "r") as f:
        content = f.read()
    import re
    content = re.sub(r'region:\s*\w+', f'region: {data.region}', content)
    content = re.sub(r'start_date:\s*".*"', f'start_date: "{data.start_date}"', content)
    content = re.sub(r'end_date:\s*".*"', f'end_date: "{data.end_date}"', content)
    content = re.sub(r'max_images:\s*\d+', f'max_images: {data.max_images}', content)
    with open(CONFIG_PATH, "w") as f:
        f.write(content)
    return {"status": "success"}

@app.post("/api/run")
def run_pipeline(background_tasks: BackgroundTasks):
    global pipeline_status
    if pipeline_status["state"] == "running":
        raise HTTPException(status_code=400, detail="Pipeline already running")
    
    background_tasks.add_task(run_pipeline_task)
    return {"status": "started"}

@app.post("/api/upload_firms")
async def upload_firms(file: UploadFile = File(...)):
    import zipfile
    upload_dir = "stage_010_data_ingestion/data/user_uploaded_firms"
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    if file.filename.endswith(".zip"):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                if member.endswith('.csv'):
                    filename = os.path.basename(member)
                    source = zip_ref.open(member)
                    target = open(os.path.join(upload_dir, filename), "wb")
                    with source, target:
                        shutil.copyfileobj(source, target)
        os.remove(file_path) # remove the zip after extracting
        
    return {"status": "success", "filename": file.filename}

@app.get("/api/uploaded_files")
def get_uploaded_files():
    upload_dir = "stage_010_data_ingestion/data/user_uploaded_firms"
    if not os.path.exists(upload_dir):
        return {"files": []}
    
    files_info = []
    for f in os.listdir(upload_dir):
        if f.endswith(".csv"):
            path = os.path.join(upload_dir, f)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            preview = []
            try:
                with open(path, 'r', encoding='utf-8') as file:
                    preview = [next(file).strip() for _ in range(2)]
            except StopIteration:
                pass
            except Exception:
                preview = ["Error reading preview"]
                
            files_info.append({
                "name": f,
                "size_mb": round(size_mb, 2),
                "preview": preview
            })
            
    return {"files": files_info}

@app.post("/api/clear_files")
def clear_uploaded_files():
    upload_dir = "stage_010_data_ingestion/data/user_uploaded_firms"
    if os.path.exists(upload_dir):
        for f in os.listdir(upload_dir):
            if f.endswith(".csv"):
                os.remove(os.path.join(upload_dir, f))
    return {"status": "success"}

@app.post("/api/resume")
def resume_pipeline(background_tasks: BackgroundTasks):
    global pipeline_status
    if pipeline_status["state"] == "running":
        raise HTTPException(status_code=400, detail="Pipeline already running")
        
    targets = []
    if not os.path.exists("stage_010_data_ingestion/data/events.csv"):
        targets.extend(["fetch", "filter", "prescreen", "build", "report"])
    elif not os.path.exists("stage_015_urban_filtering/data/events_urban.csv"):
        targets.extend(["filter", "prescreen", "build", "report"])
    elif not os.path.exists("stage_020_prescreening/data/events_prescreened.csv"):
        targets.extend(["prescreen", "build", "report"])
    else:
        # Build is naturally resumable (it checks processed patches)
        targets.extend(["build", "report"])
        
    background_tasks.add_task(run_custom_pipeline_task, targets)
    return {"status": "started", "resume_targets": targets}

@app.get("/api/status")
def get_status():
    return pipeline_status

@app.get("/api/logs")
def get_logs():
    if not os.path.exists(LOG_FILE):
        return {"logs": ""}
    with open(LOG_FILE, "r") as f:
        return {"logs": f.read()}

@app.post("/api/stop")
def stop_pipeline():
    global pipeline_status
    if pipeline_status["state"] in ["running", "paused"] and pipeline_status["pid"]:
        try:
            os.killpg(os.getpgid(pipeline_status["pid"]), signal.SIGKILL)
        except ProcessLookupError:
            pass
        pipeline_status["state"] = "idle"
        pipeline_status["pid"] = None
        return {"status": "stopped"}
    return {"status": "not running"}

@app.post("/api/pause")
def pause_pipeline():
    global pipeline_status
    if pipeline_status["state"] == "running" and pipeline_status["pid"]:
        try:
            os.killpg(os.getpgid(pipeline_status["pid"]), signal.SIGSTOP)
            pipeline_status["state"] = "paused"
            return {"status": "paused"}
        except ProcessLookupError:
            pass
    return {"status": "not running"}

@app.post("/api/resume_paused")
def resume_paused_pipeline():
    global pipeline_status
    if pipeline_status["state"] == "paused" and pipeline_status["pid"]:
        try:
            os.killpg(os.getpgid(pipeline_status["pid"]), signal.SIGCONT)
            pipeline_status["state"] = "running"
            return {"status": "resumed"}
        except ProcessLookupError:
            pass
    return {"status": "not paused"}
