import spidev
import os
import time
import threading
import cv2
import matplotlib.pyplot as plt
import numpy as np
import socket
import joblib
import subprocess
import requests
from picamera2 import Picamera2
from PIL import Image, ImageDraw, ImageFont
from gpiozero import DigitalOutputDevice
from NIRS import NIRS
from io import BytesIO
from datetime import datetime
from queue import Queue
from pyzbar.pyzbar import decode

# GPIO and SPI setup
DC_PIN = 24
RST_PIN = 25
dc = DigitalOutputDevice(DC_PIN)
rst = DigitalOutputDevice(RST_PIN)

spi_display = spidev.SpiDev()
spi_display.open(0, 0)
spi_display.max_speed_hz = 64000000

spi_adc = spidev.SpiDev()
spi_adc.open(1, 0)
spi_adc.max_speed_hz = 1000000

# Button thresholds for ADC values
BUTTON_THRESHOLDS = {
    "down": 10,
    "left": 45,
    "right": 100,
    "up": 175,
    "select": 370
}

# Global variables
picam2 = Picamera2()
camera_active = False
button_pressed = False
button_queue = []
image_queue = Queue(maxsize=1)
frame_state = "1"
last_base_img = "UI/1.jpg"
last_displayed_image = None
API_URL = "https://carefully-real-narwhal.ngrok-free.app"
wifi_connected_image = "wifi/wifi.jpg"
wifi_disconnected_image = "wifi/nowifi.jpg"

# Basic SPI and Display Functions
def read_adc_channel(channel):
    """Reads MCP3208 ADC value from the specified channel."""
    if not (0 <= channel <= 7):
        raise ValueError("Channel must be between 0 and 7")
    adc = spi_adc.xfer2([1, (8 + channel) << 4, 0])
    return ((adc[1] & 3) << 8) + adc[2]

def read_button():
    """Checks for button press and returns the button name if pressed."""
    global button_pressed
    value = read_adc_channel(1)
    for button, threshold in BUTTON_THRESHOLDS.items():
        if value < threshold and not button_pressed:
            button_pressed = True
            return button
    if value >= BUTTON_THRESHOLDS["select"]:
        button_pressed = False
    return None

def send_display_command(cmd):
    dc.off()
    spi_display.xfer([cmd])

def send_display_data(data):
    dc.on()
    spi_display.xfer([data])

def reset_display():
    """Resets the display hardware."""
    rst.off()
    time.sleep(0.5)
    rst.on()
    time.sleep(0.5)

def init_display():
    """Initializes the display with necessary commands."""
    reset_display()
    send_display_command(0x01)  # Software reset
    time.sleep(0.5)
    send_display_command(0x11)  # Exit sleep mode
    time.sleep(0.12)
    send_display_command(0x3A)  # Interface Pixel Format
    send_display_data(0x55)     # 16 bits per pixel
    send_display_command(0x36)  # Memory Access Control
    send_display_data(0x48)     # MX, BGR mode
    send_display_command(0x29)  # Display ON

def send_image_to_display(pixels):
    """Sends pixel data to the display."""
    send_display_command(0x2C)
    dc.on()  # Data mode
    pixel_data = bytearray()
    for r, g, b in pixels:
        color = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
        pixel_data.extend([(color >> 8) & 0xFF, color & 0xFF])
    for i in range(0, len(pixel_data), 4096):
        spi_display.xfer(pixel_data[i:i + 4096])

def display_image(image_path):
    """Loads and displays an image on the screen."""
    global last_displayed_image
    img = Image.open(image_path) if isinstance(image_path, str) else image_path
    img = img.resize((240, 320))  # Screen resolution
    pixels = list(img.getdata())
    if pixels != last_displayed_image:
        send_image_to_display(pixels)
        last_displayed_image = pixels

def edit_image(base_image, overlays=None, texts=None):
    """Edits an image with overlays and text."""
    img = Image.open(base_image) if isinstance(base_image, str) else base_image.copy()
    if overlays:
        for overlay_path, position, size in overlays:
            overlay = Image.open(overlay_path) if isinstance(overlay_path, str) else overlay_path
            overlay = overlay.resize(size) if size else overlay
            img.paste(overlay, position, overlay if overlay.mode == 'RGBA' else None)
    if texts:
        draw = ImageDraw.Draw(img)
        for text, position, size, color in texts:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
            draw.text(position, text, font=font, fill=color)
    return img

# NIR Scanner and Plotting
def plot_wavelength_data(wavelength_data, intensity_data):
    """Plots wavelength vs intensity and returns as an image."""
    fig, ax = plt.subplots(facecolor='#c3c4bf')
    ax.set_facecolor('#b0b0ac')
    ax.plot(wavelength_data, intensity_data, color='blue')
    ax.set_ylabel('Intensity')
    ax.set_xlabel('Wavelength')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    img = img.resize((240, 320))
    plt.close(fig)
    return img

# Camera Handling
def start_camera_feed():
    """Starts the camera feed and puts frames in a queue."""
    global camera_active
    picam2.configure(picam2.create_still_configuration(main={"size": (720, 1280), "format": "RGB888"}))
    picam2.start()
    while camera_active:
        frame = cv2.cvtColor(picam2.capture_array(), cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        if image_queue.full():
            image_queue.get()
        image_queue.put(img)
        time.sleep(0.2)

def stop_camera_feed():
    """Stops the camera feed."""
    global camera_active
    camera_active = False
    picam2.stop()

# Network and WiFi functions
def is_wifi_connected():
    """Checks if WiFi is connected."""
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=2)
        return True
    except OSError:
        return False

def connect_to_wifi(ssid, password):
    """Connects to a WiFi network with given SSID and password."""
    command = f'nmcli dev wifi connect "{ssid}" password "{password}"'
    subprocess.run(command, shell=True, check=True)

# UI Updating
def update_status_bar(base_image_path):
    """Updates the status bar with time and WiFi status."""
    current_wifi = wifi_connected_image if is_wifi_connected() else wifi_disconnected_image
    current_time = datetime.now().strftime("%H:%M")
    overlays = [(current_wifi, (410, 15), (50, 50))]
    texts = [(current_time, (170, 20), 40, "white")]
    edited_image = edit_image(base_image_path, overlays=overlays, texts=texts)
    display_image(edited_image)

# Button Listener for UI Navigation
def button_listener():
    """Listens for button presses and navigates the UI accordingly."""
    global frame_state, camera_active, image_queue, last_camera_feed, img_rp1, img_rp2
    while True:
        button = read_button()
        if button:
            button_queue.append(button)

        if frame_state == "1":
            update_status_bar("UI/1.jpg")
            if button_queue:
                if button_queue[0] == 'down':
                    frame_state = "2"
                elif button_queue[0] in ['right', 'select']:
                    frame_state = "1_1"
                button_queue.pop(0)

        elif frame_state == "2":
            update_status_bar("UI/2.jpg")
            if button_queue:
                if button_queue[0] == 'down':
                    frame_state = "3"
                    display_image("UI/3.jpg")
                elif button_queue[0] == 'up':
                    frame_state = "1"
                    display_image("UI/1.jpg")
                elif button_queue[0] in ['right', 'select']:
                    download_firmware()
                button_queue.pop(0)

        elif frame_state == "3":
            update_status_bar("UI/3.jpg")
            if button_queue:
                if button_queue[0] == 'up':
                    frame_state = "2"
                    display_image("UI/2.jpg")
                elif button_queue[0] in ['right', 'select']:
                    frame_state = "3_1"
                    camera_active = True
                    threading.Thread(target=start_camera_feed).start()
                button_queue.pop(0)

        elif frame_state == "1_1":
            update_status_bar("UI/1_1.jpg")
            if button_queue:
                if button_queue[0] in ['right', 'select']:
                    frame_state = "1_1_1"
                    display_image("UI/1_1_1.jpg")
                elif button_queue[0] == 'left':
                    frame_state = "1"
                    display_image("UI/1.jpg")
                elif button_queue[0] == 'down':
                    frame_state = "1_2"
                    display_image("UI/1_2.jpg")
                button_queue.pop(0)

        elif frame_state == "1_2":
            update_status_bar("UI/1_2.jpg")
            if button_queue:
                if button_queue[0] in ['right', 'select']:
                    frame_state = "1_1_1"
                    display_image("UI/1_1_1.jpg")
                elif button_queue[0] == 'left':
                    frame_state = "1"
                    display_image("UI/1.jpg")
                elif button_queue[0] == 'up':
                    frame_state = "1_1"
                    display_image("UI/1_1.jpg")
                button_queue.pop(0)

        elif frame_state == "1_1_1":
            update_status_bar("UI/1_1_1.jpg")
            if button_queue:
                nirs = NIRS()
                if button_queue[0] in ['right', 'select']:
                    nirs.clear_error_status()
                    nirs.scan()
                    results = nirs.get_scan_results()
                    intensity = results["intensity"]
                    reference = results["reference"]
                    absorbance = np.log10(np.array(intensity) / np.array(reference))
                    model = joblib.load('svm_model.pkl')
                    prediction = model.predict(absorbance.reshape(1, -1))
                    payload = {
                        "labelId": int(prediction[0]),
                        "systemTemp": results["temperature_system"],
                        "detectorTemp": results["temperature_detector"],
                        "humidity": results["humidity"],
                        "absorbance": absorbance.tolist(),
                        "referenceSignal": reference,
                        "sampleSignal": intensity
                    }
                    upload_data(payload)
                    if prediction == [1]:  # Fresh
                        frame_state = "1_1_1_1"
                        display_image("UI/1_1_1_1.jpg")
                    else:  # Not Fresh
                        frame_state = "1_1_1_2"
                        display_image("UI/1_1_1_2.jpg")
                elif button_queue[0] == 'left':
                    frame_state = "1_1"
                    display_image("UI/1_1.jpg")
                elif button_queue[0] == 'up':
                    nirs.set_lamp_on_off(1)
                elif button_queue[0] == 'down':
                    nirs.set_lamp_on_off(0)
                button_queue.pop(0)

        elif frame_state == "1_1_1_1":
            update_status_bar("UI/1_1_1_1.jpg")
            if button_queue:
                if button_queue[0] in ['right', 'select', 'left']:
                    frame_state = "1_1_1"
                    display_image("UI/1_1_1.jpg")
                button_queue.pop(0)

        elif frame_state == "1_1_1_2":
            update_status_bar("UI/1_1_1_2.jpg")
            if button_queue:
                if button_queue[0] in ['right', 'select', 'left']:
                    frame_state = "1_1_1"
                    display_image("UI/1_1_1.jpg")
                elif button_queue[0] == 'down':
                    frame_state = "1_1_1_3"
                    display_image("UI/1_1_1_3.jpg")
                button_queue.pop(0)

        elif frame_state == "1_1_1_3":
            update_status_bar("UI/1_1_1_3.jpg")
            if button_queue:
                if button_queue[0] in ['right', 'select']:
                    frame_state = "1_1_1_3_1"
                    camera_active = True
                    threading.Thread(target=start_camera_feed).start()
                elif button_queue[0] == 'left':
                    frame_state = "1_1_1_2"
                    display_image("UI/1_1_1_2.jpg")
                elif button_queue[0] == 'up':
                    frame_state = "1_1_1_2"
                    display_image("UI/1_1_1_2.jpg")
                button_queue.pop(0)

        elif frame_state == "1_1_1_3_1":
            if button_queue:
                while image_queue.empty():
                    time.sleep(0.2)
                last_camera_feed = image_queue.get()
                display_image(last_camera_feed)
                decoded_qr_codes = decode(last_camera_feed)
                if decoded_qr_codes:
                    img_rp1 = last_camera_feed
                    frame_state = "1_1_1_3_2"
                    display_image("UI/1_1_1_3_2.jpg")
                if button_queue[0] == 'left':
                    frame_state = "1_1_1_3"
                    stop_camera_feed()
                button_queue.pop(0)

        elif frame_state == "1_1_1_3_2":
            if button_queue:
                while image_queue.empty():
                    time.sleep(0.2)
                last_camera_feed = image_queue.get()
                display_image(last_camera_feed)
                if button_queue[0] == 'left':
                    frame_state = "1_1_1_3_1"
                elif button_queue[0] in ['right', 'select']:
                    img_rp2 = last_camera_feed
                    frame_state = "1_1_1_3_3"
                    display_image("UI/1_1_1_3_3.jpg")
                button_queue.pop(0)

        elif frame_state == "1_1_1_3_3":
            if button_queue:
                display_image("UI/1_1_1_3_3.jpg")
                if button_queue[0] in ['right', 'select', 'left']:
                    frame_state = "1_1_1_3_2"
                    camera_active = True
                    threading.Thread(target=start_camera_feed).start()
                elif button_queue[0] == 'down':
                    frame_state = "1_1_1_3_4"
                    display_image("UI/1_1_1_3_4.jpg")
                button_queue.pop(0)

        elif frame_state == "1_1_1_3_4":
            if button_queue:
                if button_queue[0] in ['right', 'select']:
                    upload_data(img_rp1, img_rp2)
                    frame_state = "1_1_1_3_5"
                    display_image("UI/1_1_1_3_5.jpg")
                elif button_queue[0] == 'left':
                    frame_state = "1_1_1_3"
                    camera_active = True
                    threading.Thread(target=start_camera_feed).start()
                elif button_queue[0] == 'up':
                    frame_state = "1_1_1_3_3"
                button_queue.pop(0)

        elif frame_state == "1_1_1_3_5":
            update_status_bar("UI/1_1_1_3_5.jpg")
            if button_queue:
                frame_state = "1_1_1_3"
                display_image("UI/1_1_1_3.jpg")
                button_queue.pop(0)

# Firmware and Data Upload
def download_firmware():
    """Downloads the latest firmware from the server."""
    url = f'{API_URL}/api/v1/PklModel/latest'
    response = requests.get(url)
    if response.status_code == 200:
        with open('latest_firmware.pkl', 'wb') as f:
            f.write(bytes(response.json()['data']['data']))

def upload_data(data):
    """Uploads intensity and metadata to the server."""
    url = f'{API_URL}/api/v1/intensities'
    requests.post(url, json=data)

# System Initialization
def initialize_system():
    """Initializes display and starts the button listener thread."""
    init_display()
    display_image("UI/0.jpg")
    threading.Thread(target=button_listener, daemon=True).start()

# Main program entry
if __name__ == "__main__":
    initialize_system()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_camera_feed()
