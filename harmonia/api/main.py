"""FastAPI application factory."""

from fastapi import FastAPI

from harmonia.api.routers import generate, jobs

app = FastAPI(title="Harmonia", version="0.1.0")

app.include_router(generate.router, prefix="/api/v1")
app.include_router(jobs.router, prefix="/api/v1")
