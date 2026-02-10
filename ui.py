import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000/summarize"

def summarize_ui(url, title):
    r = requests.post(
        API_URL,
        json={"url": url, "title": title},
        timeout=120
    )

    if r.status_code != 200:
        return f"ERROR: {r.text}"

    return r.json()["summary"]

gr_app = gr.Interface(
    fn=summarize_ui,
    title="AI Article Summarizer",
    inputs=[
        gr.Textbox(label="Article URL",info="Enter a Valid URl"),
        gr.Textbox(label="Article Title",info="Enter article Title")
    ],
    outputs=gr.Textbox(label="Summary", lines=15),
    examples=[
        ["https://pmc.ncbi.nlm.nih.gov/articles/PMC3936971/" , "Understanding logistic regression analysis"],
        ["https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8099726","Densely Connected Convolutional Networks"]
    ],
    flagging_mode='never'
)
