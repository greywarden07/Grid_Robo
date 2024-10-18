import cv2
import io
import os
import re
import csv
import time
from google.cloud import vision
from datetime import datetime
from collections import Counter

# Set the environment variable for Google Application Credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gen-lang-client-0751335714-ca33e8fe888b.json"

# Set up the Google Cloud Vision client
client = vision.ImageAnnotatorClient()

# Function to capture a frame and send it to Google Vision API
def detect_text_from_frame(frame):
    try:
        # Convert the captured frame to bytes
        _, encoded_image = cv2.imencode('.jpg', frame)
        content = encoded_image.tobytes()

        # Create an image object for the Vision API
        image = vision.Image(content=content)

        # Use the Vision API to detect text
        response = client.text_detection(image=image)
        texts = response.text_annotations

        # Extract and return detected text if any
        if texts:
            return texts[0].description.strip()
    except Exception as e:
        print(f"Error detecting text: {e}")

    return None

# Expanded regex pattern for a variety of expiry date formats
expiry_pattern = r"(?:\b(?:Exp(?:iry)?[: ]?)?(?:\d{2}[\/\-]\d{2}[\/\-]\d{4})|" \
                 r"(?:\d{2}[\/\-]\d{4})|" \
                 r"(?:\d{4}[\/\-]\d{2})|" \
                 r"(?:\b(?:Exp(?:iry)?[: ]?)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[ ]?\d{4})|" \
                 r"(?:\d{2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{4}))"

# Function to validate and extract specific product information
def extract_product_details(text):
    product_details = {
        'Brand': '',
        'Pack Size': '',
        'MRP': '',
        'Expiry Date': ''
    }

    # Expanded regex for MRP, expiry date, and pack size
    mrp_pattern = r"(₹|Rs\.?|INR)[ ]?\d{1,3}(?:,\d{3})*(\.\d{1,2})?"
    pack_size_pattern = r"(\d+(\.\d+)?\s?(g|kg|mg|ml|L|litre|liters|gm|grams|cc|cubic|oz|ounces))"

    # Validate and extract MRP
    mrp_match = re.search(mrp_pattern, text)
    if mrp_match:
        product_details['MRP'] = mrp_match.group()

    # Validate and extract expiry date
    expiry_match = re.search(expiry_pattern, text, re.IGNORECASE)
    if expiry_match:
        expiry_date_str = expiry_match.group()
        try:
            # Handle various formats for the expiry date
            if "/" in expiry_date_str or "-" in expiry_date_str:
                # Handle dates like DD/MM/YYYY or MM/YYYY
                if len(expiry_date_str.split('/')[0]) == 2:  # Check if it's DD/MM/YYYY
                    expiry_date = datetime.strptime(expiry_date_str, "%d/%m/%Y")
                else:  # MM/YYYY
                    expiry_date = datetime.strptime(expiry_date_str, "%m/%Y")
            else:
                # Handle formats like Jan 2024 or 20 Jan 2024
                expiry_date = datetime.strptime(expiry_date_str, "%b %Y")
                
            # Ensure expiry date is in the future
            if expiry_date > datetime.now():
                product_details['Expiry Date'] = expiry_date_str
        except ValueError:
            pass  # Invalid date format, leave empty

    # Validate and extract pack size
    pack_size_match = re.search(pack_size_pattern, text, re.IGNORECASE)
    if pack_size_match:
        product_details['Pack Size'] = pack_size_match.group()

    # Validate and extract brand (Assume brand is first line without numbers or special characters)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    for line in lines:
        # Expanded brand detection: allow alphanumeric and limited special characters (like apostrophes or hyphens)
        if re.match(r"^[a-zA-Z0-9\s'&.-]+$", line):
            product_details['Brand'] = line
            break

    return product_details

# CSV Writing Setup
csv_file_path = 'product_details.csv'
csv_headers = ['Brand', 'Pack Size', 'MRP', 'Expiry Date']

# Open CSV in append mode
with open(csv_file_path, 'a', newline='', encoding='utf-8') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
    
    # Write the header only if the file is empty
    if csv_file.tell() == 0:
        writer.writeheader()

    # Initialize the camera
    cap = cv2.VideoCapture(0)  # 0 is the default camera, change it if needed

    # Capture frames from the camera in real-time
    while True:
        start_time = time.time()
        brand_counter = Counter()
        pack_size_counter = Counter()
        mrp_counter = Counter()
        expiry_date_counter = Counter()

        while time.time() - start_time < 10:  # Scan for 10 seconds
            ret, frame = cap.read()  # Capture a frame
            if not ret:
                print("Failed to capture image")
                break

            # Display the frame
            cv2.imshow('Camera Feed', frame)

            # Detect text from the current frame
            detected_text = detect_text_from_frame(frame)
            if detected_text:
                # Extract product details with validation
                product_details = extract_product_details(detected_text)

                # Update counters with detected product details
                if product_details['Brand']:
                    brand_counter[product_details['Brand']] += 1
                if product_details['Pack Size']:
                    pack_size_counter[product_details['Pack Size']] += 1
                if product_details['MRP']:
                    mrp_counter[product_details['MRP']] += 1
                if product_details['Expiry Date']:
                    expiry_date_counter[product_details['Expiry Date']] += 1

                # Print the detected product details for debugging
                print("Product Details:", product_details)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Determine the most common product details
        most_common_brand = brand_counter.most_common(1)[0][0] if brand_counter else ''
        most_common_pack_size = pack_size_counter.most_common(1)[0][0] if pack_size_counter else ''
        most_common_mrp = mrp_counter.most_common(1)[0][0] if mrp_counter else ''
        most_common_expiry_date = expiry_date_counter.most_common(1)[0][0] if expiry_date_counter else ''

        # Write the most common product details to CSV even if some fields are missing
        if any([most_common_brand, most_common_pack_size, most_common_mrp, most_common_expiry_date]):
            writer.writerow({
                'Brand': most_common_brand,
                'Pack Size': most_common_pack_size,
                'MRP': most_common_mrp,
                'Expiry Date': most_common_expiry_date
            })
            print(f"Written to CSV: Brand={most_common_brand}, Pack Size={most_common_pack_size}, MRP={most_common_mrp}, Expiry Date={most_common_expiry_date}")

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
