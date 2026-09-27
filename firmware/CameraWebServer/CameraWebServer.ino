#include <Arduino.h>
#include <WiFi.h>
#include "esp_camera.h"
#include "esp_http_server.h"

#define CAMERA_AP_SSID "LaneTalk-Camera"
#define CAMERA_AP_PASSWORD "YOUR_CAMERA_AP_PASSWORD"

// AI-Thinker ESP32-CAM pin map (OV2640).
#define CAMERA_MODEL_AI_THINKER
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22

static const char *STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=frame";
static httpd_handle_t cameraServer = nullptr;
constexpr int BUTTON_PIN = 13;
static int lastRawButtonState = LOW;

static esp_err_t indexHandler(httpd_req_t *request) {
  httpd_resp_set_type(request, "text/plain; charset=utf-8");
  return httpd_resp_sendstr(request, "LaneTalk camera ready. MJPEG stream: /stream\n");
}

static esp_err_t streamHandler(httpd_req_t *request) {
  httpd_resp_set_type(request, STREAM_CONTENT_TYPE);
  httpd_resp_set_hdr(request, "Cache-Control", "no-cache");

  Serial.println("[http] MJPEG client connected");
  char partHeader[96];

  while (true) {
    camera_fb_t *frame = esp_camera_fb_get();
    if (frame == nullptr) {
      Serial.println("[camera] Frame capture failed");
      return ESP_FAIL;
    }

    const int headerLength = snprintf(
        partHeader, sizeof(partHeader),
        "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n",
        static_cast<unsigned int>(frame->len));

    esp_err_t result = httpd_resp_send_chunk(request, partHeader, headerLength);
    if (result == ESP_OK) {
      result = httpd_resp_send_chunk(request, reinterpret_cast<const char *>(frame->buf), frame->len);
    }
    if (result == ESP_OK) {
      result = httpd_resp_send_chunk(request, "\r\n", 2);
    }

    esp_camera_fb_return(frame);
    if (result != ESP_OK) {
      Serial.println("[http] MJPEG client disconnected");
      break;
    }
  }

  return ESP_FAIL;
}

static bool startCameraServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.max_uri_handlers = 2;

  esp_err_t result = httpd_start(&cameraServer, &config);
  if (result != ESP_OK) {
    Serial.printf("[http] Server start failed: %s (0x%x)\n", esp_err_to_name(result), result);
    return false;
  }

  httpd_uri_t indexUri = {};
  indexUri.uri = "/";
  indexUri.method = HTTP_GET;
  indexUri.handler = indexHandler;

  httpd_uri_t streamUri = {};
  streamUri.uri = "/stream";
  streamUri.method = HTTP_GET;
  streamUri.handler = streamHandler;

  result = httpd_register_uri_handler(cameraServer, &indexUri);
  if (result == ESP_OK) {
    result = httpd_register_uri_handler(cameraServer, &streamUri);
  }
  if (result != ESP_OK) {
    Serial.printf("[http] Route registration failed: %s (0x%x)\n", esp_err_to_name(result), result);
    httpd_stop(cameraServer);
    cameraServer = nullptr;
    return false;
  }

  Serial.println("[http] Server listening on port 80");
  return true;
}

static void pollButtonRawState() {
  const int rawState = digitalRead(BUTTON_PIN);
  if (rawState == lastRawButtonState) {
    return;
  }

  lastRawButtonState = rawState;
  Serial.println(rawState == HIGH ? "[button] raw=HIGH" : "[button] raw=LOW");
}

static void startCameraAccessPoint() {
  WiFi.mode(WIFI_AP);
  WiFi.setSleep(false);
  if (!WiFi.softAP(CAMERA_AP_SSID, CAMERA_AP_PASSWORD)) {
    Serial.println("[wifi] SoftAP start failed");
    while (true) {
      delay(1000);
    }
  }
  Serial.printf("[wifi] SoftAP started: %s\n", CAMERA_AP_SSID);
  Serial.printf("[wifi] Camera AP IP: %s\n", WiFi.softAPIP().toString().c_str());
}

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n[boot] LaneTalk AI-Thinker ESP32-CAM");

  pinMode(BUTTON_PIN, INPUT);
  lastRawButtonState = digitalRead(BUTTON_PIN);
  Serial.printf("[button] GPIO=%d\n", BUTTON_PIN);
  Serial.printf("[button] initial_state=%s\n",
                lastRawButtonState == HIGH ? "HIGH" : "LOW");

  const bool hasPsram = psramFound();
  Serial.printf("[camera] PSRAM: %s\n", hasPsram ? "available" : "not found");

  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = hasPsram ? FRAMESIZE_VGA : FRAMESIZE_QVGA;
  config.jpeg_quality = 12;
  config.fb_count = hasPsram ? 2 : 1;
  config.fb_location = hasPsram ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;
  config.grab_mode = hasPsram ? CAMERA_GRAB_LATEST : CAMERA_GRAB_WHEN_EMPTY;

  const esp_err_t cameraResult = esp_camera_init(&config);
  if (cameraResult != ESP_OK) {
    Serial.printf("[camera] Initialization failed: %s (0x%x)\n",
                  esp_err_to_name(cameraResult), cameraResult);
    Serial.println("[camera] Check the OV2640 ribbon cable, camera board, and stable 5V power.");
    while (true) {
      delay(1000);
    }
  }
  Serial.println("[camera] OV2640 initialized");

  startCameraAccessPoint();

  if (!startCameraServer()) {
    Serial.println("[http] Could not start the camera server; stopping.");
    while (true) {
      delay(1000);
    }
  }

  Serial.printf("[ready] Open http://%s/stream\n", WiFi.softAPIP().toString().c_str());
}

void loop() {
  pollButtonRawState();
  delay(1);
}
