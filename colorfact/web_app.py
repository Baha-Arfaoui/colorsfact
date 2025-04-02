import gradio as gr
from gradio.utils import abspath
import os
import time
from langchain_community.llms import Ollama
from imageAnalysis import ImageAnalysis
import yaml
import base64
from io import BytesIO
import json
from PIL import Image
from pathlib import Path
from pyprojroot import here
from fastapi.staticfiles import StaticFiles
from utils import extract_colors_image,extract_colors_url,matching_products,load_data_outfit_mapping
import pandas as pd
# Configuration
image_folder = 'data/images'

image_analysis = ImageAnalysis()
selected_image = None
# Load outfit recommendations from YAML file

load=load_data_outfit_mapping(data_path="data/output.xlsx",outfit_path="configs/outfit.yaml")
# Load outfit recommendations
outfit_config = load.load_outfit()
data=load.load_dataset()
# # Load available product images
# data=pd.read_excel("data/final_output.xlsx",index_col=0) 
# data['Genre']=data["Genre"].apply(lambda x : x.strip())
# outfit_path="configs/outfit.yaml"
# with open(outfit_path,encoding="utf-8") as cfg:
#             outfit_config = yaml.load(cfg, Loader=yaml.FullLoader)

# Helper functions
def on_select_gallery_image(evt: gr.SelectData):
    global selected_image
    selected_image = evt.value['image']['path']
    print("on_select_gallery_image: " + selected_image)
    return selected_image

def analyze_image(image_path):
    if image_path is None:
        return {"error": "No image selected"}, None, None, None
    
    # Analyze the image
    try:
        rulesParameters, processed_image,colors = image_analysis.display_analysis(image_path)
        
        # In a real implementation, you would detect the product type from the image
        # For demonstration, we'll just use a simple detection that selects a random product type
        import random
        product_types = list(outfit_config.get("Outfit", {}).keys())
        if not product_types:
            return {"error": "No product types found in configuration"}, processed_image, None, None
        
        # Simulating detection - in a real app, this would come from your image analysis
        params = rulesParameters.get('extractedPictureElements', {})
        detected_product = params.get('Product', 'Unknown')
        
        # Get recommendations for the detected product
        recommendations_outfit = outfit_config.get("Outfit", {}).get(detected_product, {})
        rec=image_analysis.get_recommendation()
        
        return rulesParameters, processed_image, colors,detected_product,recommendations_outfit,rec
    except Exception as e:
        return {"error": f"Error analyzing image: {str(e)}"}, None, None, None, None,None

def format_analysis_results(analysis):
    if "error" in analysis:
        return analysis["error"]
    
    params = analysis.get('extractedPictureElements', {})
    result = f"""
        ### Image Analysis
        - Product: {params.get('Product', 'Unknown')}
        - Gender: {params.get('Gender', 'Unknown')}
            """
    return result




# Get images for gallery
image_paths = [os.path.join(image_folder, file) for file in os.listdir(image_folder)
               if file.lower().endswith(('png', 'jpg', 'jpeg'))]

# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Row():
        gr.HTML("""
        <div style="text-align: center; margin-bottom: 10px;">
            <h1>Outfit Recommendation System</h1>
            <p>Upload or select an image of a clothing item to get personalized outfit recommendations</p>
        </div>
        """)
    
    # Input section
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("<h3>Select or upload an image</h3>")
            upload_image = gr.Image(type="filepath", label="Upload Image", height=300)
            
            with gr.Accordion("Or choose from gallery", open=False):
                gallery = gr.Gallery(
                    label="Gallery", 
                    value=image_paths,
                    columns=4,
                    object_fit="cover",
                    height=250
                )
            
            analyze_btn = gr.Button("Get Outfit Recommendations", variant="primary", size="lg")
        
        # Results section
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("Analysis"):
                    with gr.Row():
                        processed_image = gr.Image(label="Processed Image", height=300)
                        analysis_results = gr.Markdown(label="Analysis Results")
                        colors=gr.Image(type='pil',label="Dominant Colors",height=100)
                                
                with gr.TabItem("Outfit Recommendations"):
                    detected_product = gr.Textbox(label="Detected Product", visible=True)
                    outfit_container = gr.HTML()
    
    # Event handlers
    gallery.select(on_select_gallery_image, None, upload_image)
    
    def process_and_display(image_path):
        if image_path is None:
            return None, "No image selected", "", """<div style="padding:20px; text-align:center;">
                <p>Please select or upload an image to get recommendations</p>
            </div>"""
        
        analysis, processed_img,colors, product_type ,recommendations_dict,rec= analyze_image(image_path)
        analysis_md = format_analysis_results(analysis)
        
        if product_type is None:
            return processed_img, analysis_md,colors, "", """<div style="padding:20px; text-align:center;">
                <p>No product type could be detected from the image</p>
            </div>"""
        
        # Format recommendations as HTML
        html_output = f"""<div style="padding:10px;">
            <div style="background-color:#f0f0f0; padding:10px; border-radius:5px; margin-bottom:20px;">
                <h2 style="margin:0;">Recommendations for: {product_type}</h2>
            </div>
        """
        
        for category, items in recommendations_dict.items():
            html_output += f"""
            <div style="margin-bottom:30px;">
                <h3 style="border-bottom:1px solid #ddd; padding-bottom:5px;">{category}</h3>
                <div style="display:flex; flex-wrap:wrap; gap:15px; margin-top:10px;">
            """
            
            for item in items:
                # In a real app, you'd get the actual product image
                # Check if product image exists
                
                # Get product image path
                try:
                    link=rec[category][item][0]["product_id"]
                except:
                    link=f"/api/placeholder/150/150"
                
                html_output += f"""
                <div style="width:150px; text-align:center; background-color:white; border-radius:8px; padding:10px; box-shadow:0 2px 4px rgba(0,0,0,0.1);">
                    <img src="{link}" alt="{item}" style="max-width:100%; height:120px; object-fit:cover; border-radius:5px;">
                    <p style="margin-top:8px; font-weight:bold;">{item}</p>
                </div>
                """
            
            html_output += """
                </div>
            </div>
            """
        
        html_output += "</div>"
        return processed_img, analysis_md,colors, product_type, gr.HTML(html_output)
    
    analyze_btn.click(
        process_and_display,
        inputs=[upload_image],
        outputs=[processed_image, analysis_results,colors, detected_product, outfit_container]
    )
    
    gr.HTML("""
    <div style="text-align:center; margin-top:20px; padding:10px; background-color:#f7f7f7; border-radius:5px;">
        <p><small>© 2025 Outfit Recommendation System - CBTW</small></p>
    </div>
    """)

if __name__ == "__main__":
    demo.launch(
        allowed_paths=["./data/images/"],
        server_name="0.0.0.0",
        show_api=False,
        favicon_path=(os.path.join(os.path.dirname(__file__), "avatar.png"))
    )

