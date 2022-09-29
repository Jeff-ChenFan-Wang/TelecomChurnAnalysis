from fastapi import FastAPI

app = FastAPI(title="TelecomChurn")

@app.get("/")
async def root():
    return {"message": "Hello World"}