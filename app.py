"""Run the same Gradio app locally as on the Hugging Face Space."""

from space.app import build_demo

if __name__ == "__main__":
    build_demo().launch()
