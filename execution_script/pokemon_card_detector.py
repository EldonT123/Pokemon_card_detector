import cv2
import pytesseract
from ultralytics import YOLO
import logging
import requests
import os
from tkinter import Tk, filedialog
import re
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading
import time
import requests
from PIL import Image
# ------------------- CONFIG -------------------
POKEMON_API_KEY = "API_KEY_HERE"
POKEMON_API_URL = "https://api.pokemontcg.io/v2/cards"
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Update if needed
MODEL_PATH = "best.pt"
OUTPUT_FILE = "ocr_results.txt"
# ------------------------------------------------

# Tesseract setup
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Logging setup
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

# Load YOLO model
logging.info("Loading YOLO model...")
model = YOLO(MODEL_PATH)
logging.info("Model loaded successfully.")

# Function: crop bounding box & OCR
def crop_and_ocr(img, box, label):
    logging.info(f"Cropping for label: {label}")
    x1, y1, x2, y2 = map(int, box)
    crop = img[y1:y2, x1:x2]
    text = pytesseract.image_to_string(crop).strip()

    if label.lower() == "card_number":
        text = re.sub(r'[^A-Za-z0-9/]', '', text)

    logging.info(f"OCR result for {label}: {text}")
    return text

# Function: save OCR results
def save_results(name, card_number):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"Name: {name}\n")
        f.write(f"Card Number: {card_number}\n")
    logging.info(f"OCR results saved to {OUTPUT_FILE}")

# Function: query Pokémon TCG API
def fetch_card_details(name, card_number, retries=3, delay=2):
    logging.info(f"Querying Pokémon TCG API for '{name}' - '{card_number}'")
    headers = {"X-Api-Key": POKEMON_API_KEY}
    card_num_only = card_number.split('/')[0] if '/' in card_number else card_number
    query = f'name:"{name}" number:{card_num_only}'

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(
                POKEMON_API_URL,
                headers=headers,
                params={"q": query},
                timeout=10  # seconds
            )
            response.raise_for_status()
            data = response.json()
            if data.get("data"):
                card = data["data"][0]
                logging.info(f"Found card: {card['name']} ({card['id']})")
                return card
            else:
                logging.warning("No matching card found.")
                return None
        except requests.exceptions.Timeout:
            logging.warning(f"Timeout on attempt {attempt} of {retries}. Retrying in {delay}s...")
            time.sleep(delay)
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}")
            return None

    logging.error("Failed all retries for API request.")
    return None

# Function: run detection, OCR, API, and display
def process_image(img):
    logging.info("Running YOLO detection...")
    results = model(img)

    # Make window resizable
    cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)

    annotated_frame = results[0].plot()

    # Optionally resize to a fixed width while maintaining aspect ratio
    max_width = 1200
    height, width = annotated_frame.shape[:2]
    if width > max_width:
        scale = max_width / width
        new_dim = (max_width, int(height * scale))
        annotated_frame = cv2.resize(annotated_frame, new_dim, interpolation=cv2.INTER_AREA)

    cv2.imshow("Detections", annotated_frame)
    cv2.waitKey(1)  # Display bounding boxes

    name_text = ""
    number_text = ""

    for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
        label = results[0].names[int(cls)]
        text = crop_and_ocr(img, box, label)
        if label.lower() == "name":
            name_text = text
        elif label.lower() == "card_number":
            number_text = text

    save_results(name_text, number_text)

    # Fetch API details if both found
    if name_text and number_text:
        logging.info("Both Name and Number found. Proceeding to API lookup...")
        details = fetch_card_details(name_text, number_text)
        if details:
            logging.info(f"API Lookup Success: {details['name']} ({details['set']['name']})")
            # Show popup in separate thread to avoid hanging
            show_card_popup(details, annotated_frame)

        else:
            logging.warning("No matching card found for OCR results.")
    else:
        logging.warning("Missing Name or Number from OCR. Skipping API lookup.")


def format_card_details(details):
    """Return a nicely formatted string from the API response dict."""
    if not isinstance(details, dict):
        return str(details)
    lines = []
    lines.append(f"Name: {details.get('name', 'N/A')}")
    set_info = details.get('set', {})
    lines.append(f"Set: {set_info.get('name', 'N/A')}")
    lines.append(f"Release Date: {set_info.get('releaseDate', 'N/A')}")
    lines.append(f"Rarity: {details.get('rarity', 'N/A')}")
    images = details.get('images', {})
    if images.get('small'):
        lines.append(f"Image URL: {images.get('small')}")
    tcgplayer = details.get('tcgplayer', {})
    prices = tcgplayer.get('prices', {})
    holo = prices.get('holofoil', {})
    if holo:
        lines.append(f"Market Price: {holo.get('market', 'N/A')}")
    return "\n".join(lines)


def show_card_popup(details, frame):
    """
    Display a Tk window with the annotated image and card details.
    Must be called on the main thread (which your script currently does).
    """
    try:
        # Create Tk root
        root = tk.Tk()
        root.title("Pokémon Card Details")

        # Convert OpenCV image (BGR) -> PIL Image (RGB)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Optionally resize image to fit comfortably in popup
        max_w, max_h = 500, 700
        w, h = pil_img.size
        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1.0:
            new_size = (int(w * scale), int(h * scale))
            pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)

        # Create PhotoImage bound to this root (master=root)
        img_tk = ImageTk.PhotoImage(pil_img, master=root)

        # Image label (keep reference to prevent GC)
        img_label = tk.Label(root, image=img_tk)
        img_label.image = img_tk
        img_label.pack(padx=8, pady=8)

        # Details text
        details_text = format_card_details(details)
        text_label = tk.Label(root, text=details_text, justify="left", anchor="w",
                              font=("Arial", 11), padx=8, pady=8)
        text_label.pack(fill="both", expand=True)

        # Close button
        close_btn = tk.Button(root, text="Close", command=root.destroy)
        close_btn.pack(pady=(0,8))

        # Run the popup (this will block until closed)
        root.mainloop()

    except Exception:
        logging.exception("Failed to open card popup")





# Main webcam loop
cap = cv2.VideoCapture(0)
logging.info("Starting webcam feed... Press [SPACE] to capture, [P] to pick an image, [ESC] to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        logging.error("Failed to capture frame from webcam.")
        break

    cv2.imshow("YOLO Live Feed", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        logging.info("Exiting program.")
        break

    elif key == 32:  # SPACEBAR
        logging.info("Spacebar pressed - capturing current frame.")
        process_image(frame)

    elif key == ord('p'):  # P key
        logging.info("P pressed - opening file dialog.")
        cap.release()
        root = Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        root.update()
        cap = cv2.VideoCapture(0)

        if file_path:
            logging.info(f"Selected image: {file_path}")
            img = cv2.imread(file_path)
            process_image(img)

cap.release()
cv2.destroyAllWindows()

