from typing import Optional
import gradio as gr
import numpy as np
import torch
from PIL import Image
import io
import base64
import os
from util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
from io import BytesIO

# Initialize models
yolo_model = get_yolo_model(model_path='weights/icon_detect/model.pt')
caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="weights/icon_caption_florence")

MARKDOWN = """
# OmniParser for Pure Vision Based General GUI Agent 🔥
<div>
    <a href="https://arxiv.org/pdf/2408.00203">
        <img src="https://img.shields.io/badge/arXiv-2408.00203-b31b1b.svg" alt="Arxiv" style="display:inline-block;">
    </a>
</div>

OmniParser is a screen parsing tool to convert general GUI screen to structured elements. 
"""

DEVICE = torch.device('cuda')

def process(
    image_input,
    box_threshold,
    iou_threshold,
    use_paddleocr,
    imgsz
) -> Optional[Image.Image]:
    image_save_path = 'imgs/saved_image_demo.png'
    image_input.save(image_save_path)
    image = Image.open(image_save_path)
    box_overlay_ratio = image.size[0] / 3200
    draw_bbox_config = {
        'text_scale': 0.8 * box_overlay_ratio,
        'text_thickness': max(int(2 * box_overlay_ratio), 1),
        'text_padding': max(int(3 * box_overlay_ratio), 1),
        'thickness': max(int(3 * box_overlay_ratio), 1),
    }

    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
        image_save_path, 
        display_img=False, 
        output_bb_format='xyxy', 
        goal_filtering=None, 
        easyocr_args={'paragraph': False, 'text_threshold':0.9}, 
        use_paddleocr=use_paddleocr
    )
    text, ocr_bbox = ocr_bbox_rslt
    
    dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
        image_save_path,
        yolo_model,
        BOX_TRESHOLD=box_threshold,
        output_coord_in_ratio=True,
        ocr_bbox=ocr_bbox,
        draw_bbox_config=draw_bbox_config,
        caption_model_processor=caption_model_processor,
        ocr_text=text,
        iou_threshold=iou_threshold,
        imgsz=imgsz,
    )
    
    image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
    print('finish processing')
    parsed_content_list = '\n'.join([f'icon {i}: ' + str(v) for i,v in enumerate(parsed_content_list)])
    return image, str(parsed_content_list)

# Create FastAPI app
app = FastAPI()

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)
    with gr.Row():
        with gr.Column():
            image_input_component = gr.Image(type='pil', label='Upload image')
            box_threshold_component = gr.Slider(
                label='Box Threshold', minimum=0.01, maximum=1.0, step=0.01, value=0.05)
            iou_threshold_component = gr.Slider(
                label='IOU Threshold', minimum=0.01, maximum=1.0, step=0.01, value=0.1)
            use_paddleocr_component = gr.Checkbox(
                label='Use PaddleOCR', value=True)
            imgsz_component = gr.Slider(
                label='Icon Detect Image Size', minimum=640, maximum=1920, step=32, value=640)
            submit_button_component = gr.Button(
                value='Submit', variant='primary')
        with gr.Column():
            image_output_component = gr.Image(type='pil', label='Image Output')
            text_output_component = gr.Textbox(label='Parsed screen elements', placeholder='Text Output')

    submit_button_component.click(
        fn=process,
        inputs=[
            image_input_component,
            box_threshold_component,
            iou_threshold_component,
            use_paddleocr_component,
            imgsz_component
        ],
        outputs=[image_output_component, text_output_component]
    )

# Mount Gradio app under /gradio path
app = gr.mount_gradio_app(app, demo, path="/gradio")

# Helper function to convert PIL Image to base64
def image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

@app.get("/")
async def root():
    return {"message": "API is running. Access Gradio interface at /gradio"}

# REST API endpoint
@app.post("/api/process_image")
async def process_image(
    file: UploadFile = File(...),
    box_threshold: float = 0.05,
    iou_threshold: float = 0.1,
    use_paddleocr: bool = True,
    imgsz: int = 640
):
    try:
        # Read and convert uploaded file to PIL Image
        image_data = await file.read()
        image = Image.open(BytesIO(image_data))
        
        # Process the image using existing function
        output_image, parsed_content = process(
            image,
            box_threshold,
            iou_threshold,
            use_paddleocr,
            imgsz
        )
        
        # Convert output image to base64
        output_image_base64 = image_to_base64(output_image)
        
        return JSONResponse({
            "status": "success",
            "output_image": output_image_base64,
            "parsed_content": parsed_content
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)