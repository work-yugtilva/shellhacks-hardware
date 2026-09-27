# LaneTalk ESP32-CAM stream

Minimal Arduino firmware for an AI-Thinker ESP32-CAM with an OV2640 camera. It starts a Wi-Fi access point, prints its IP address at 115200 baud, and serves a multipart MJPEG stream at `/stream`.

## Camera access point credentials

The ESP32-CAM starts a Wi-Fi access point. Before flashing, set a private network name and password near the top of `CameraWebServer.ino`:

```cpp
#define CAMERA_AP_SSID "YOUR_CAMERA_AP_SSID"
#define CAMERA_AP_PASSWORD "YOUR_CAMERA_AP_PASSWORD"
```

Use a password of at least 8 characters. The sketch starts the access point with these settings and prints its address to Serial Monitor.

## Arduino IDE setup

1. Install Arduino IDE.
2. In **Arduino IDE > Settings/Preferences**, add Espressif's stable Boards Manager URL:
   `https://espressif.github.io/arduino-esp32/package_esp32_index.json`
3. In **Tools > Board > Boards Manager**, install **esp32 by Espressif Systems**. Version **3.3.12** is the version used to compile this sketch.
4. Open `CameraWebServer.ino` and select **Tools > Board > esp32 > AI Thinker ESP32-CAM**.
5. Use these **Tools** settings:
   - **Upload Speed:** 115200
   - **CPU Frequency:** 240MHz (WiFi/BT)
   - **Flash Frequency:** 80MHz
   - **Flash Mode:** QIO
   - **Partition Scheme:** Huge APP (3MB No OTA/1MB SPIFFS)
   - **Core Debug Level:** None
   - **Erase All Flash Before Sketch Upload:** Disabled

The AI-Thinker board profile enables its PSRAM; no separate PSRAM menu option is needed. The sketch checks for PSRAM and uses VGA when present, QVGA if absent.

## FT232RL wiring and upload

Your signal wiring is right for UART0 flashing: TXD to U0R/RX, RXD to U0T/TX, GND to GND, and GPIO0 to GND while entering the bootloader. Check these two electrical details:

- Both UART directions must be 3.3 V logic. Set the FT232RL board's I/O level (VCCIO) to 3.3 V; do not feed 5 V TXD into U0R. If the adapter cannot provide 3.3 V logic, use a level shifter on both UART signal lines.
- The board's 5V input needs a stable supply. The FT232RL's onboard 3.3 V output is not a camera power source (the FT232R datasheet allows up to 50 mA from that output). If the FT232 5V rail causes resets or camera failures, power ESP32-CAM 5V from a regulated 5V/1A supply, keep the grounds common, and disconnect the FT232 5V wire so two 5V sources are not tied together. Espressif recommends a 3.3 V rail capable of at least 500 mA for the ESP32 chip; the camera adds load.

Upload steps:

1. Wire the FT232 and hold **GPIO0 to GND**.
2. Connect the FT232 to the laptop. In Arduino IDE, select its serial port under **Tools > Port**.
3. Click **Upload**. After upload completes, disconnect GPIO0 from GND and press the ESP32-CAM **RST** button (or briefly power-cycle it) to run the sketch normally.
4. Open **Tools > Serial Monitor** at **115200 baud**. Watch for `[wifi] SoftAP started: ...`, `[wifi] Camera AP IP: ...`, and `[ready] Open http://.../stream`.

If upload stays on **Connecting...**:

1. Confirm GPIO0 is connected to GND, then tap **RST** while the IDE is trying to connect.
2. Confirm the right serial port is selected, close Serial Monitor, and check crossed TX/RX plus common GND.
3. Keep Upload Speed at 115200 and check that the FT232 TXD is 3.3 V logic and that 5V power is stable.

## Stream check

After flashing, remove the GPIO0-to-GND jumper, reset the board, and wait for the AP IP in Serial Monitor. Connect the laptop to the camera access point using the SSID and password set above, then open:

```text
http://<ESP32-CAM-IP>/stream
```

The browser should show the live camera feed. `http://<ESP32-CAM-IP>/` returns a short status line naming `/stream`. This is an unauthenticated local-network stream; keep it on a trusted network.

## Next laptop-side step

Keep the camera transport behind a small interface so detection and speech do not depend on how frames arrive:

```text
laptop/
  requirements.txt       # opencv-python, requests (if using the explicit MJPEG reader)
  .env.example           # CAMERA_STREAM_URL=http://<ip>/stream
  lanetalk/
    camera.py            # CameraSource.read() -> BGR frame; reconnect on stream loss
    events.py             # Event(kind, confidence, timestamp, details)
    detector.py           # detect(frame) -> list[Event]; initially a stub
    coach.py              # event(s) -> concise spoken text
    speech.py             # say(text); TTS adapter behind this function
    app.py                # frame loop -> detect -> coach -> speech
```

First implement `CameraSource.read()` with `cv2.VideoCapture(CAMERA_STREAM_URL)` and verify it returns OpenCV BGR frames; if that OpenCV build does not decode multipart MJPEG, read the HTTP response with `requests` and decode each JPEG using `cv2.imdecode`. Then add detectors that emit timestamped events, map those events to short non-repeating coach lines, and pass lines to a laptop text-to-speech backend. Keep frame acquisition, event detection, coach policy, and speech output as separate interfaces.

## Verification

Compiled successfully with Arduino-ESP32 3.3.12 for FQBN `esp32:esp32:esp32cam` (AI Thinker ESP32-CAM). No firmware has been uploaded.
