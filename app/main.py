import uvicorn

if __name__ == "__main__":
    ## todo :: more workers
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=True)