from fastapi import FastAPI, Request, Form
import uvicorn
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

app = FastAPI(title="TelecomChurn")

BASE_PATH = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(BASE_PATH / "dashboard"))


@app.get("/")
async def root(request: Request) -> dict:
    result = 0
    return TEMPLATES.TemplateResponse(
        "index.html",
        {"request": request,'result': result},
    )

@app.post("/")
def form_post(request: Request, num: int = Form(...)):
    result = str(num)
    return TEMPLATES.TemplateResponse(
        'index.html', 
        context={'request': request, 'result': result}
    )

    
@app.get("/eda")
async def root(request: Request) -> dict:
    return TEMPLATES.TemplateResponse(
        "EDA.html",
        {"request": request},
    )
    
@app.get("/model")
async def root(request: Request) -> dict:
    return TEMPLATES.TemplateResponse(
        "model.html",
        {"request": request},
    )

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)