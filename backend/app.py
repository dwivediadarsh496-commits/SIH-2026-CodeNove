import gradio as gr
from main import app as fastapi_app

# Create a clean UI placeholder while exposing full FastAPI endpoints
with gr.Blocks(title="GeoSR-AI API Backend") as demo:
    gr.Markdown("# 🛰️ GeoSR-AI Backend API is Running!")
    gr.Markdown("FastAPI backend is active and listening for requests from Next.js frontend.")
    gr.Markdown("Interactive API docs are available at: `/api/docs`")

# Mount Gradio into FastAPI
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
