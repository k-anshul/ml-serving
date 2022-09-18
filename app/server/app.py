from fastapi import FastAPI

from server.routes.ml_routes import router as ModelRouter
from concurrent.futures.process import ProcessPoolExecutor
import asyncio

app = FastAPI()

app.include_router(ModelRouter, tags=["Models"], prefix="/app")

@app.on_event("startup")
async def startup_event():
    app.state.executor = ProcessPoolExecutor()


@app.on_event("shutdown")
async def on_shutdown():
    app.state.executor.shutdown()