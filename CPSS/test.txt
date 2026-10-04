#include <WiFi.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <Wire.h>
#include <Adafruit_ADS1X15.h>

// ── WiFi ───────────────────────────────────────────────────
const char* ssid     = "San";
const char* password = "Sanjit@2007";

// ── WebSocket  (run ipconfig every boot, update this IP) ───
const char* FRIEND_PC_IP = "10.69.14.192";   // <-- UPDATE EACH BOOT
const int   WS_PORT      = 8000;
const char* WS_PATH      = "/ws";

// ── Pins ───────────────────────────────────────────────────
#define TEMP_PIN 4
#define TURB_PIN 35

// ── Objects ────────────────────────────────────────────────
OneWire           oneWire(TEMP_PIN);
DallasTemperature sensors(&oneWire);
WebSocketsClient  webSocket;
Adafruit_ADS1115  ads;

bool wsConnected  = false;
bool wifiWasDown  = false;

// ── WebSocket Events ───────────────────────────────────────
void onWebSocketEvent(WStype_t type, uint8_t* payload, size_t length) {
  switch (type) {
    case WStype_CONNECTED:
      wsConnected = true;
      Serial.println("[WS] Connected ✓");
      break;
    case WStype_DISCONNECTED:
      wsConnected = false;
      Serial.println("[WS] Disconnected — retrying...");
      break;
    case WStype_TEXT:
      Serial.printf("[WS] Server: %s\n", payload);
      break;
    case WStype_ERROR:
      Serial.println("[WS] Error");
      break;
    default:
      break;
  }
}

// ── WiFi (non-blocking with timeout + WebSocket restart) ───
void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) {
    wifiWasDown = false;
    return;
  }

  wifiWasDown = true;
  Serial.print("Connecting WiFi");
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi OK → " + WiFi.localIP().toString());
    // Restart WebSocket after WiFi reconnect
    webSocket.begin(FRIEND_PC_IP, WS_PORT, WS_PATH);
    webSocket.setReconnectInterval(3000);
    webSocket.enableHeartbeat(15000, 3000, 2);
  } else {
    Serial.println("\nWiFi FAILED — will retry");
  }
}

// ── Stable ADC Read ────────────────────────────────────────
float getStableRead(int pin) {
  float sum = 0;
  for (int i = 0; i < 10; i++) {
    sum += analogRead(pin);
    delay(10);
  }
  return sum / 10.0;
}

// ── TDS via ADS1115 (channel 0) ────────────────────────────
float readTDS(float tempC) {
  float sum = 0;
  for (int i = 0; i < 10; i++) { sum += ads.readADC_SingleEnded(0); delay(10); }
  float volts  = (sum / 10.0) * 0.1875 / 1000.0;
  float rawTDS = (133.42 * pow(volts, 3)) - (255.86 * pow(volts, 2)) + (857.39 * volts);
  float comp   = 1.0 + 0.02 * (tempC - 25.0);
  return (rawTDS / comp) * 0.5;
}

// ── Turbidity via GPIO 35 ──────────────────────────────────
float readTurbidityVolts() {
  return getStableRead(TURB_PIN) * (3.3 / 4095.0);
}

float convertTurbidityToNTU(float volts) {
  return (-434.78 * volts) + 1434.78;
}

// ── pH via ADS1115 (channel 1) ─────────────────────────────
float readPH(float tempC) {
  float sum = 0;
  for (int i = 0; i < 10; i++) { sum += ads.readADC_SingleEnded(1); delay(10); }
  float volts = (sum / 10.0) * 0.1875 / 1000.0;
  float ph    = 7.0 + ((2.5 - volts) / 0.18);
  ph         -= 0.03 * (tempC - 25.0);
  return constrain(ph, 0.0, 14.0);
}

// ── Send JSON via WebSocket ────────────────────────────────
void sendSensorData(float temp, float tds, float turbidityNTU, float ph) {
  StaticJsonDocument<256> doc;
  doc["Temp"]      = serialized(String(temp,         2));
  doc["TDS"]       = serialized(String(tds,          2));
  doc["Turbidity"] = serialized(String(turbidityNTU, 2));
  doc["pH"]        = serialized(String(ph,           2));

  char buf[256];
  serializeJson(doc, buf);
  webSocket.sendTXT(buf);
  Serial.printf("[WS] Sent: %s\n", buf);
}

// ── Setup ──────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(1000);

  analogReadResolution(12);
  analogSetPinAttenuation(TURB_PIN, ADC_11db);

  sensors.begin();

  Wire.begin(21, 22);
  if (!ads.begin()) {
    Serial.println("ADS1115 NOT FOUND — check wiring!");
    while (1);
  }

  connectWiFi();

  // Register event handler BEFORE begin()
  webSocket.onEvent(onWebSocketEvent);
  webSocket.begin(FRIEND_PC_IP, WS_PORT, WS_PATH);
  webSocket.setReconnectInterval(3000);
  webSocket.enableHeartbeat(15000, 3000, 2);

  Serial.println("SYSTEM READY");
}

// ── Loop ───────────────────────────────────────────────────
void loop() {
  connectWiFi();
  webSocket.loop();

  static unsigned long lastSend = 0;
  if (millis() - lastSend >= 5000) {
    lastSend = millis();

    sensors.requestTemperatures();
    float temp         = sensors.getTempCByIndex(0);
    float tds          = readTDS(temp);
    float turbVolts    = readTurbidityVolts();
    float turbidityNTU = convertTurbidityToNTU(turbVolts);
    float ph           = readPH(temp);

    Serial.println("================================");
    if (temp == -127.00 || temp == 85.00)
      Serial.println("TEMP:  !! SENSOR ERROR !!");
    else
      Serial.printf("TEMP:  %.2f C\n", temp);
    Serial.printf("TDS:   %.2f ppm\n",  tds);
    Serial.printf("TURB:  %.2f NTU\n", turbidityNTU);
    Serial.printf("pH:    %.2f\n",       ph);
    Serial.printf("WS:    %s\n",         wsConnected ? "CONNECTED ✓" : "WAITING...");
    Serial.println("================================");

    if (wsConnected) {
      sendSensorData(temp, tds, turbidityNTU, ph);
    } else {
      Serial.println("Not connected — skipping send.");
    }
  }
}