from fastapi import FastAPI
import gradio as gr
from api import app as fastapi_app
from ui import gr_app

app = fastapi_app

app = gr.mount_gradio_app(
    app,
    gr_app,
    path="/"
)
