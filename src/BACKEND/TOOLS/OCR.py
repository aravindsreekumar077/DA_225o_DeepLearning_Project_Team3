from fastapi.responses import FileResponse
import uuid
import cv2
import pytesseract
import os

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
def get_ocr_text(image):

    try:
        # Save the uploaded image
        filename = f"{uuid.uuid4().hex}.jpg"
        input_path = os.path.join(UPLOAD_FOLDER, filename)

        with open(input_path, "wb") as f:
            f.write( image)

        # Load and process the image
        img = cv2.imread(input_path)
        if img is None:
            raise HTTPException(status_code=400, detail="Failed to read image")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(thresh, config=custom_config)

        rows = text.strip().split('\n')
        output_filename = f"{uuid.uuid4().hex}.txt"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        with open(output_path, "w") as f:
            for item in rows:
                f.write(f"{item}\n")

        return FileResponse(path=output_path, filename=output_filename, media_type='text/plain')


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

