import requests
import base64
from PIL import Image
from io import BytesIO
import pytest

API_URL = "http://localhost:8000/process_image"

def test_process_image():
    # Load a test image
    test_image_path = "imgs/saved_image_demo.png"
    
    # Prepare the files and data for the request
    files = {
        'file': ('test_image.png', open(test_image_path, 'rb'), 'image/png')
    }
    
    data = {
        'box_threshold': 0.05,
        'iou_threshold': 0.1,
        'use_paddleocr': True,
        'imgsz': 640
    }
    
    try:
        print("Processing image, please wait...")
        # Make the POST request
        response = requests.post(API_URL, files=files, data=data)
        
        # Check if request was successful
        assert response.status_code == 200, f"Request failed with status code {response.status_code}"
        
        # Parse the response
        result = response.json()
        
        # Basic validation of response structure
        assert 'status' in result, "Response missing 'status' field"
        assert 'output_image' in result, "Response missing 'output_image' field"
        assert 'parsed_content' in result, "Response missing 'parsed_content' field"
        assert result['status'] == 'success', "Processing was not successful"
        
        # Validate the output image
        image_data = base64.b64decode(result['output_image'])
        output_image = Image.open(BytesIO(image_data))
        assert isinstance(output_image, Image.Image), "Output image is not a valid image"
        
        # Validate parsed content
        assert isinstance(result['parsed_content'], str), "Parsed content is not a string"
        assert len(result['parsed_content']) > 0, "Parsed content is empty"
        
        print("✅ Test passed successfully!")
        print("Parsed content preview:", result['parsed_content'])
        
        # Optionally save the output image for manual inspection
        output_image.save("imgs/test_output.png")
        print("Output image saved as 'test_output.png'")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        raise
    finally:
        # Clean up
        files['file'][1].close()

if __name__ == "__main__":
    test_process_image()