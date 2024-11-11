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
import psutil
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
lamp_on_1_image = "lamp/lamp_on_1.jpg"
lamp_on_0_image = "lamp/lamp_on_0.jpg"
lamp_off_1_image = "lamp/lamp_off_1.jpg"
lamp_off_0_image = "lamp/lamp_off_0.jpg"
beef_model = "svm_model.pkl"
pork_model = "xgb_model.pkl"

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
    chunk_size = 4096
    for i in range(0, len(pixel_data), chunk_size):
        spi_display.xfer(pixel_data[i:i + chunk_size])

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
        time.sleep(0.2)  # Control FPS

def stop_camera_feed():
    """Stops the camera feed."""
    global camera_active
    camera_active = False
    # if camera_thread and camera_thread.is_alive():
    #     camera_thread.join()  # Wait for the camera thread to end
    picam2.stop()

# Network and WiFi functions
def is_wifi_connected():
    """Checks if WiFi is connected."""
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=2)
        return True
    except OSError:
        return False

# def connect_to_wifi(ssid, password):
#     """Connects to a WiFi network with given SSID and password."""
#     command = f'nmcli dev wifi connect "{ssid}" password "{password}"'
#     subprocess.run(command, shell=True, check=True)

# def parse_wifi_info(qr_data):
#     """Get SSID and password information from QR data."""
#     if qr_data.startswith("WIFI:"):
#         qr_data = qr_data[5:].strip()
#         details = qr_data.split(";")
#         ssid = details[0].split(":")[1]
#         password = details[1].split(":")[1]
#         return ssid, password
#     else:
#         # raise ValueError("QR code format not supported")
#         return False

def connect_wifi_from_qr(image):
    qrcodes = decode(image)
    if qrcodes:
        for qr in qrcodes:
            wifi_info = qr.data.decode("utf-8")
            if wifi_info.startswith("WIFI:"):
                wifi_info = wifi_info[5:].strip()
                details = wifi_info.split(";")
                ssid = details[0].split(":")[1]
                password = details[1].split(":")[1]
                command = f'nmcli dev wifi connect "{ssid}" password "{password}"'
                subprocess.run(command, shell=True, check=True)

# Firmware and Data Upload
# def download_firmware():
#     """Downloads the latest firmware from the server."""
#     url = f'{API_URL}/api/v1/PklModel/latest'
#     response = requests.get(url)
#     if response.status_code == 200:
#         file_path = os.path.join(os.getcwd(), 'output_model.pkl')
#         with open(file_path, 'wb') as f:
#             f.write(bytes(response.json()['data']['data']))
#         return True
#     else:
#         return False
def download_firmware():
    url = f'{API_URL}/api/v1/PklModel/latest'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        buffer_value_svm = bytes(data['data'][0]['data'])
        buffer_value_xgb = bytes(data['data'][1]['data'])
        file_path_svm = os.path.join(os.getcwd(), 'svm_model.pkl')
        file_path_xgb = os.path.join(os.getcwd(), 'xgb_model.pkl')
        with open(file_path_svm, 'wb') as f:
            f.write(buffer_value_svm)
        with open(file_path_xgb, 'wb') as f:
            f.write(buffer_value_xgb)
        return True
    else:
        return False

def upload_data(data):
    """Uploads intensity and metadata to the server."""
    url = f'{API_URL}/api/v1/intensities'
    requests.post(url, json=data)

def upload_report(qr_img, proof_img):
    """Uploads QR and food images while reporting to the server."""
    url = f'{API_URL}/api/v1/report/upload'
    files = {
        'file1': ('image1.jpg', qr_img, 'image/jpeg'),
        'file2': ('image2.jpg', proof_img, 'image/jpeg')
    }
    requests.post(url, files=files)

# UI Updating
def update_status_bar(base_image_path):
    """Updates the status bar with time and WiFi status."""
    temp_cpu = psutil.sensors_temperatures()['cpu_thermal'][0].current
    temp_cpu = f"{temp_cpu:.1f}°C"
    nirs = NIRS()
    temp_detector = nirs.get_detector_temperature()
    temp_detector = f"{temp_detector:.1f}°C"
    current_time = datetime.now().strftime("%H:%M")
    current_wifi = wifi_connected_image if is_wifi_connected() else wifi_disconnected_image
    overlays = [(current_wifi, (410, 15), (50, 50))]
    texts = [(temp_cpu, (10, 10), 15, "#F4F7F5"),
        (temp_detector, (10, 30), 15, "#F4F7F5"),
        (current_time, (170, 20), 40, "#F4F7F5")]
    edited_image = edit_image(base_image_path, overlays=overlays, texts=texts)
    display_image(edited_image)

def update_status_bar_wLamp(base_image, type=None):
    """Updates the status bar and lamp ON/OFF"""
    nirs = NIRS()
    if type:
        current_lamp = lamp_on_1_image if nirs.get_lamp_status() else lamp_off_1_image
    else:
        current_lamp = lamp_on_0_image if nirs.get_lamp_status() else lamp_off_0_image
    overlays = [(current_lamp, (400, 100), (100, 100))]
    edited_image = edit_image(base_image, overlays=overlays)
    update_status_bar(edited_image)

def update_status_bar_wLamp_wResult(base_image, wavelength_data, intensity_data, type=None):
    """Updates the status bar and lamp ON/OFF and NIR Scan result"""
    edited_image = edit_image(base_image, overlays=[(plot_wavelength_data(wavelength_data, intensity_data), (5, 205), (505, 430))])
    update_status_bar_wLamp(edited_image, type) if type else update_status_bar_wLamp(edited_image)

def update_status_bar_wCamFeed(base_image, camera_feed, wifi_name=None):
    """Updates the status bar and camera feed and wifi name if available"""
    edited_image = edit_image(base_image, overlays=[(last_camera_feed, (15, 205), (450, 420))])
    if wifi_name:
        if is_wifi_connected():
            result = subprocess.run(["iwgetid", "-r"], capture_output=True, text=True, check=True)
            ssid = result.stdout.strip()
        else:
            ssid = "Wifi disconnected!"
        edited_image = edit_image(edited_image, texts = [(ssid, (25, 105), 35, "#DAA520")])
    update_status_bar(edited_image)

# Button Listener for UI Navigation
def button_listener():
    """Listens for button presses and navigates the UI accordingly."""
    global frame_state, camera_active, image_queue, last_camera_feed, img_rp1, img_rp2
    while True:
        button = read_button()
        if button:
            button_queue.append(button)

        match frame_state:
            case "1":
                update_status_bar("UI/1.jpg")
                if button_queue:
                    if button_queue[0] == 'down':
                        frame_state = "2"
                    elif button_queue[0] == 'select':
                        if os.path.exists("model_beef.pkl") and os.path.exists("model_pork.pkl"):
                            frame_state = "1_1"
                        else:
                            frame_state = "1_3"
                    button_queue.pop(0)

            case "1_1":
                update_status_bar("UI/1_1.jpg")
                if button_queue:
                    if button_queue[0] == 'select':
                        current_model = beef_model
                        frame_state = "1_1_1"
                    elif button_queue[0] == 'left':
                        frame_state = "1"
                    elif button_queue[0] == 'down':
                        frame_state = "1_2"
                    button_queue.pop(0)
            case "1_2":
                update_status_bar("UI/1_2.jpg")
                if button_queue:
                    if button_queue[0] == 'select':
                        current_model = pork_model
                        frame_state = "1_1_1"
                    elif button_queue[0] == 'left':
                        frame_state = "1"
                    elif button_queue[0] == 'up':
                        frame_state = "1_1"
                    button_queue.pop(0)
            case "1_3":
                update_status_bar("UI/1_3.jpg")
                if button_queue:
                    frame_state = "1"
                    button_queue.pop(0)

            case "1_1_1":
                nirs = NIRS()
                update_status_bar_wLamp("UI/1_1_1.jpg")
                if button_queue:
                    if button_queue[0] == 'select':
                        nirs.scan()
                        results = nirs.get_scan_results()
                        intensity = results["intensity"]
                        # reference = results["reference"]
                        reference = [70933, 80262, 88283, 96274, 109441, 121705, 136827, 153952, 172041, 189436, 205751, 220790, 234985, 
                            248851, 262772, 276941, 295696, 309995, 324162, 337919, 350632, 362296, 372938, 382790, 390847, 397725, 
                            403039, 407212, 411373, 414042, 415732, 416463, 417887, 419439, 419376, 418787, 417571, 416250, 415476, 
                            415824, 415636, 414934, 415258, 416688, 418092, 418577, 419049, 420626, 421921, 423485, 424547, 426392, 
                            428216, 429421, 430229, 429856, 430102, 431299, 431558, 431869, 431408, 431071, 431211, 431043, 431309, 
                            432018, 432699, 433864, 435151, 436698, 438349, 440047, 441859, 444372, 446593, 449517, 453344, 460494, 
                            465942, 471436, 477387, 484137, 491504, 499443, 507992, 517337, 526381, 535099, 544160, 557692, 566366, 
                            574549, 583273, 593536, 602595, 610610, 620041, 630517, 641673, 652903, 663974, 678083, 686919, 695836, 
                            703672, 710976, 717196, 723574, 730209, 734312, 737269, 740676, 744394, 748484, 749402, 750310, 751569, 
                            752321, 750432, 747000, 743065, 738146, 732502, 727901, 724493, 720250, 716099, 712015, 707971, 704182, 
                            700601, 697263, 693919, 691367, 689450, 687141, 681895, 673733, 665255, 665164, 667200, 666720, 664126, 
                            661717, 659641, 658158, 656205, 653983, 652078, 650148, 646661, 643794, 640373, 636362, 632823, 629454, 
                            626148, 622781, 619825, 616343, 612129, 608315, 604603, 601736, 598566, 595026, 591114, 587318, 583365, 
                            579318, 574636, 569072, 563656, 558535, 551301, 544753, 538261, 532374, 526656, 520005, 513107, 506614, 
                            499861, 492540, 485365, 478179, 468432, 460535, 452929, 445517, 437608, 429768, 422103, 415453, 408069, 
                            399981, 392229, 385104, 374962, 366739, 358752, 350719, 342888, 334192, 325445, 317355, 309538, 301884, 
                            293823, 286199, 275650, 267564, 259244, 250549, 241122, 231273, 220846, 209165, 195554, 179442, 161643, 
                            142514, 116244, 97730, 81527, 68306, 57601, 48540]
                        absorbance = np.log10(np.array(intensity) / np.array(reference))
                        model = joblib.load(current_model)
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
                            frame_state = "1_1_1_11"
                        else:  # Not Fresh
                            frame_state = "1_1_1_21"
                    elif button_queue[0] == 'left':
                        frame_state = "1_1"
                    elif button_queue[0] == 'right':
                        frame_state = "1_1_2"
                    button_queue.pop(0)

            case "1_1_1_11":
                update_status_bar_wLamp_wResult("UI/1_1_1_11.jpg", results["wavelength"], results["intensity"])
                if button_queue:
                    if button_queue[0] == 'select':
                        button_queue.append('select')
                        frame_state = "1_1_1"
                    elif button_queue[0] == 'right':
                        frame_state = "1_1_1_12"
                    elif button_queue[0] == 'left':
                        frame_state = "1_1_1"
                    button_queue.pop(0)
            case "1_1_1_12":
                update_status_bar_wLamp_wResult("UI/1_1_1_12.jpg", results["wavelength"], results["intensity"], True)
                if button_queue:
                    if button_queue[0] == 'select':
                        nirs.set_lamp_on_off(0) if nirs.get_lamp_status() else nirs.set_lamp_on_off(1)
                    elif button_queue[0] == 'down':
                        frame_state = "1_1_1_13"
                    elif button_queue[0] == 'left':
                        frame_state = "1_1_1_11"
                    button_queue.pop(0)
            case "1_1_1_13":
                update_status_bar_wLamp_wResult("UI/1_1_1_13.jpg", results["wavelength"], results["intensity"])
                if button_queue:
                    if button_queue[0] == 'select':
                        nirs.clear_error_status()
                    elif button_queue[0] == 'up':
                        frame_state = "1_1_1_12"
                    elif button_queue[0] == 'left':
                        frame_state = "1_1_1"
                    button_queue.pop(0)
            case "1_1_1_21":
                update_status_bar_wLamp_wResult("UI/1_1_1_21.jpg", results["wavelength"], results["intensity"])
                if button_queue:
                    if button_queue[0] == 'select':
                        button_queue.append('select')
                        frame_state = "1_1_1"
                    elif button_queue[0] == 'right':
                        frame_state = "1_1_1_22"
                    elif button_queue[0] == 'left':
                        frame_state = "1_1_1"
                    elif button_queue[0] == 'down':
                        frame_state = "1_1_1_24"
                    button_queue.pop(0)
            case "1_1_1_22":
                update_status_bar_wLamp_wResult("UI/1_1_1_22.jpg", results["wavelength"], results["intensity"], True)
                if button_queue:
                    if button_queue[0] == 'select':
                        nirs.set_lamp_on_off(0) if nirs.get_lamp_status() else nirs.set_lamp_on_off(1)
                    elif button_queue[0] == 'down':
                        frame_state = "1_1_1_23"
                    elif button_queue[0] == 'left':
                        frame_state = "1_1_1_21"
                    button_queue.pop(0)
            case "1_1_1_23":
                update_status_bar_wLamp_wResult("UI/1_1_1_23.jpg", results["wavelength"], results["intensity"])
                if button_queue:
                    if button_queue[0] == 'select':
                        nirs.clear_error_status()
                    elif button_queue[0] == 'up':
                        frame_state = "1_1_1_12"
                    elif button_queue[0] == 'left':
                        frame_state = "1_1_1_24"
                    button_queue.pop(0)
            case "1_1_1_24":
                update_status_bar_wLamp_wResult("UI/1_1_1_24.jpg", results["wavelength"], results["intensity"])
                if button_queue:
                    if button_queue[0] == 'select':
                        camera_active = True
                        threading.Thread(target=start_camera_feed).start()
                        # camera_thread = threading.Thread(target=start_camera_feed)
                        # camera_thread.start()
                        frame_state = "1_1_1_x_1"
                    elif button_queue[0] == 'up':
                        frame_state = "1_1_1_11"
                    elif button_queue[0] == 'right':
                        frame_state = "1_1_1_23"
                    button_queue.pop(0)

            case "1_1_1_x_1":
                while image_queue.empty():
                    time.sleep(0.2)
                last_camera_feed = image_queue.get()
                update_status_bar_wCamFeed("1_1_1_x_1", last_camera_feed)
                decoded_qr_codes = decode(last_camera_feed)
                if decoded_qr_codes:
                    img_rp1 = last_camera_feed
                    frame_state = "1_1_1_x_2"
                if button_queue:
                    if button_queue[0] == 'left':
                        stop_camera_feed()
                        frame_state = "1_1_1_24"
                    button_queue.pop(0)
            case "1_1_1_x_2":
                while image_queue.empty():
                    time.sleep(0.2)
                last_camera_feed = image_queue.get()
                update_status_bar_wCamFeed("1_1_1_x_2", last_camera_feed)
                if button_queue:
                    if button_queue[0] == 'select':
                        stop_camera_feed()
                        frame_state = "1_1_1_x_3"
                    elif button_queue[0] == 'left':
                        frame_state = "1_1_1_x_1"
                    button_queue.pop(0)
            case "1_1_1_x_3":
                update_status_bar_wCamFeed("1_1_1_x_3", last_camera_feed)
                if button_queue:
                    if button_queue[0] == 'select':
                        frame_state = "1_1_1_x_5"
                    elif button_queue[0] == 'left':
                        camera_active = True
                        threading.Thread(target=start_camera_feed).start()
                        # camera_thread = threading.Thread(target=start_camera_feed)
                        # camera_thread.start()
                        frame_state = "1_1_1_x_2"
                    elif button_queue[0] == 'down':
                        frame_state = "1_1_1_x_4"
                    button_queue.pop(0)
            case "1_1_1_x_4":
                update_status_bar_wCamFeed("1_1_1_x_4", last_camera_feed)
                if button_queue:
                    if button_queue[0] in ['select', 'left']:
                        camera_active = True
                        threading.Thread(target=start_camera_feed).start()
                        # camera_thread = threading.Thread(target=start_camera_feed)
                        # camera_thread.start()
                        frame_state = "1_1_1_x_2"
                    elif button_queue[0] == 'up':
                        frame_state = "1_1_1_x_3"
                    button_queue.pop(0)
            case "1_1_1_x_5":
                update_status_bar("1_1_1_x_5")
                if button_queue:
                    frame_state = "1_1_1_24"
                    button_queue.pop(0)

            case "1_1_2":
                current_lamp = lamp_on_1_image if nirs.get_lamp_status() else lamp_off_1_image
                UI_for_112 = edit_image("UI/1_1_2.jpg", overlays=[(current_lamp, (400, 100), (100, 100))])
                update_status_bar(UI_for_112)
                if button_queue:
                    if button_queue[0] == 'select':
                        nirs.set_lamp_on_off(0) if nirs.get_lamp_status() else nirs.set_lamp_on_off(1)
                    elif button_queue[0] == 'left':
                        frame_state = "1_1_1"
                    elif button_queue[0] == 'down':
                        frame_state = "1_1_3"
                    button_queue.pop(0)
            case "1_1_3":
                current_lamp = lamp_on_0_image if nirs.get_lamp_status() else lamp_off_0_image
                UI_for_113 = edit_image("UI/1_1_3.jpg", overlays=[(current_lamp, (400, 100), (100, 100))])
                update_status_bar(UI_for_113)
                if button_queue:
                    if button_queue[0] == 'select':
                        nirs.clear_error_status()
                    elif button_queue[0] == 'up':
                        frame_state = "1_1_2"
                    elif button_queue[0] == 'left':
                        frame_state = "1_1"
                    button_queue.pop(0)

            case "2":
                update_status_bar("UI/2.jpg")
                if button_queue:
                    if button_queue[0] == 'select':
                        if is_wifi_connected:
                            frame_state = "2_1"
                        else:
                            frame_state = "2_4"
                    elif button_queue[0] == 'down':
                        frame_state = "3"
                    elif button_queue[0] == 'up':
                        frame_state = "1"
                    button_queue.pop(0)
            case "2_1":
                update_status_bar("UI/2_1.jpg")
                frame_state = "2_2" if download_firmware() else "2_3"
            case "2_2":
                update_status_bar("UI/2_2.jpg")
                if button_queue:
                    frame_state = "2"
                    button_queue.pop(0)
            case "2_3":
                update_status_bar("UI/2_3.jpg")
                if button_queue:
                    frame_state = "2"
                    button_queue.pop(0)
            case "2_4":
                update_status_bar("UI/2_4.jpg")
                if button_queue:
                    frame_state = "2"
                    button_queue.pop(0)
                
            case "3":
                update_status_bar("UI/3.jpg")
                if button_queue:
                    if button_queue[0] == 'select':
                        camera_active = True
                        threading.Thread(target=start_camera_feed).start()
                        frame_state = "3_1" if is_wifi_connected() else "3_2"
                    elif button_queue[0] == 'up':
                        frame_state = "2"
                    button_queue.pop(0)
            case "3_1":
                if not is_wifi_connected:
                    frame_state = "3_2"
                while image_queue.empty():
                    time.sleep(0.2)
                last_camera_feed = image_queue.get()
                update_status_bar_wCamFeed("UI/3_1.jpg", last_camera_feed, True)
                connect_wifi_from_qr(last_camera_feed)
                # qrcodes = decode(last_camera_feed)
                # if qrcodes:
                #     for qr in qrcodes:
                #         wifi_info = qr.data.decode("utf-8")
                #         ssid, password = parse_wifi_info(wifi_info)
                #         if ssid and password:
                #             connect_to_wifi(ssid, password)
                if button_queue:
                    if button_queue[0] == 'left':
                        frame_state = "3"
                    button_queue.pop(0)
            case "3_2":
                if is_wifi_connected():
                    frame_state = "3_1"
                while image_queue.empty():
                    time.sleep(0.2)
                last_camera_feed = image_queue.get()
                update_status_bar_wCamFeed("UI/3_2.jpg")
                connect_wifi_from_qr(last_camera_feed)
                # qrcodes = decode(last_camera_feed)
                # if qrcodes:
                #     for qr in qrcodes:
                #         wifi_info = qr.data.decode("utf-8")
                #         ssid, password = parse_wifi_info(wifi_info)
                #         if ssid and password:
                #             connect_to_wifi(ssid, password)
                if button_queue:
                    if button_queue[0] == 'left':
                        frame_state = "3"
                    button_queue.pop(0)

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
