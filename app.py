from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os

app = FastAPI()

# Serve static files (for output images)
app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_DIR = "uploads"
OUTPUT_IMAGE = "static/combined_mask_with_label.png"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("static", exist_ok=True)

def run_mcp_on_image(image_path):
    # Import your MCP main logic here
    # For example, you can import main from combine_vit_unet_mcp and call it
    # You may need to refactor your MCP code to accept an image path as argument
    from combine_vit_unet_mcp import main as mcp_main
    mcp_main(image_path=image_path)  # You may need to adjust this
    # Optionally, return the class label if your MCP code provides it
    # For now, just return a placeholder
    return "Class label here"

@app.get("/", response_class=HTMLResponse)
async def main_form():
    return """
    <html>
        <head>
            <title>Mould Detection Web App</title>
        </head>
        <body>
            <h2>Upload a building image for mould detection</h2>
            <form action="/upload" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept="image/*">
                <input type="submit" value="Upload">
            </form>
        </body>
    </html>
    """

@app.post("/upload", response_class=HTMLResponse)
async def upload_image(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    vit_class, vit_label = run_mcp_on_image(file_location)
    return JSONResponse({
        "vit_label": vit_label,
        "mask_url": "/static/combined_mask_with_label.png"
    })

# To run: uvicorn app:app --reload