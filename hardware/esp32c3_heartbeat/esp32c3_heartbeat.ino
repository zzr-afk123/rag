/*
 * RAG heartbeat monitor - ESP32-C3 + SSD1306 OLED + buzzer
 *
 * It is a BLE peripheral. A PC-side script (scripts/heartbeat_server.py)
 * connects to it and writes one status line whenever it polls the RAG
 * service. The board renders the line on the OLED and beeps when the
 * service goes down.
 *
 * Protocol: one UTF-8 line of key=value pairs separated by ';'
 *   state=OK;lat=123;qpm=4;chunks=90;mock=0;msg=optional text
 *   state is one of OK / WARN / DOWN
 *
 * Wiring (ESP32-C3 SuperMini / DevKitM-1):
 *   OLED VCC -> 3V3      OLED GND -> GND
 *   OLED SDA -> GPIO8    OLED SCL -> GPIO9      (I2C address 0x3C)
 *   Buzzer   -> see BUZZER_* below
 *
 * BUZZER WIRING NOTE
 *   Your buzzer has its GND plugged into the pin labelled 0 / IO0 / D0,
 *   i.e. GPIO0, with its + on 3V3. That works with BUZZER_ACTIVE_LOW=1:
 *   GPIO0 idles HIGH (silent), a beep pulls it LOW. GPIO0 is also the
 *   boot-mode strapping pin, so if flashing ever fails or the board refuses
 *   to start, unplug the buzzer from IO0, flash, then plug it back. Moving
 *   the buzzer to GPIO3 or GPIO10 with GND on a real GND pin is cleaner -
 *   then set BUZZER_ACTIVE_LOW=0.
 */

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// ------------------------------- wiring ----------------------------------
#define I2C_SDA 8
#define I2C_SCL 9
#define OLED_ADDR 0x3C
#define BUZZER_PIN 0
#define BUZZER_ACTIVE_LOW 1

// ------------------------------ protocol ---------------------------------
#define NUS_SERVICE_UUID "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
#define NUS_RX_UUID "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"  // PC -> board (write)
#define NUS_TX_UUID "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"  // board -> PC (notify)
#define BLE_NAME "RAG-HB"

// A status line older than this means the PC stopped talking to us.
#define LOST_MS 15000UL
// While the service is down, remind every N ms.
#define REMIND_MS 20000UL

Adafruit_SSD1306 display(128, 64, &Wire, -1);

BLEServer *server = nullptr;
BLECharacteristic *txChar = nullptr;
bool clientConnected = false;

// --------------------------- status received -----------------------------
String stateText = "WAIT";  // OK / WARN / DOWN
int latencyMs = 0;
int qpm = 0;
int chunks = 0;
int mock = 0;
String message = "";
unsigned long lastLineMs = 0;
bool haveLine = false;

// ------------------------------ buzzer -----------------------------------
struct BeepPlan {
  int count = 0;
  int onMs = 0;
  int offMs = 0;
  unsigned long nextAt = 0;
  bool sounding = false;
} beep;

void buzzerOff() { digitalWrite(BUZZER_PIN, BUZZER_ACTIVE_LOW ? HIGH : LOW); }
void buzzerOn() { digitalWrite(BUZZER_PIN, BUZZER_ACTIVE_LOW ? LOW : HIGH); }

// Beeps `count` times: onMs sounding, offMs silence between them.
void beepPattern(int count, int onMs, int offMs) {
  beep.count = count > 0 ? count * 2 - 1 : 0;
  beep.onMs = onMs;
  beep.offMs = offMs;
  beep.nextAt = millis();
  beep.sounding = false;
}

void serviceBuzzer() {
  if (beep.count <= 0) return;
  unsigned long now = millis();
  if (now < beep.nextAt) return;
  if (!beep.sounding) {
    buzzerOn();
    beep.sounding = true;
    beep.count--;
    beep.nextAt = now + beep.onMs;
  } else {
    buzzerOff();
    beep.sounding = false;
    beep.count--;
    beep.nextAt = now + beep.offMs;
  }
}

// ------------------------------ helpers ----------------------------------
const char *effectiveState() {
  if (!haveLine) return "WAIT";
  if (millis() - lastLineMs > LOST_MS) return "LOST";
  return stateText.c_str();
}

bool isBadState(const char *s) {
  return (strcmp(s, "DOWN") == 0) || (strcmp(s, "LOST") == 0);
}

void applyState(const char *next, const char *prev) {
  if (strcmp(next, prev) == 0) return;
  if (isBadState(next)) {
    beepPattern(3, 220, 160);
  } else if (isBadState(prev)) {
    beepPattern(2, 90, 70);
  } else if (strcmp(next, "WARN") == 0) {
    beepPattern(1, 120, 0);
  }
}

void parseLine(const String &line) {
  char prev[8];
  strncpy(prev, effectiveState(), sizeof(prev) - 1);
  prev[sizeof(prev) - 1] = '\0';

  int i = 0;
  while (i < (int)line.length()) {
    int j = line.indexOf(';', i);
    if (j < 0) j = line.length();
    String kv = line.substring(i, j);
    kv.trim();
    int eq = kv.indexOf('=');
    if (eq > 0) {
      String k = kv.substring(0, eq);
      String v = kv.substring(eq + 1);
      k.trim();
      if (k == "state") {
        v.toUpperCase();
        if (v == "OK" || v == "WARN" || v == "DOWN") stateText = v;
      } else if (k == "lat") {
        latencyMs = v.toInt();
      } else if (k == "qpm") {
        qpm = v.toInt();
      } else if (k == "chunks") {
        chunks = v.toInt();
      } else if (k == "mock") {
        mock = v.toInt();
      } else if (k == "msg") {
        message = v;
      }
    }
    i = j + 1;
  }
  lastLineMs = millis();
  haveLine = true;
  applyState(effectiveState(), prev);
}

// ------------------------------- BLE -------------------------------------
class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *s) override {
    clientConnected = true;
    beepPattern(1, 60, 0);
  }
  void onDisconnect(BLEServer *s) override {
    clientConnected = false;
    BLEDevice::startAdvertising();
  }
};

class RxCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *c) override {
    String value = c->getValue().c_str();
    value.trim();
    if (value.length() == 0) return;
    parseLine(value);
  }
};

// ------------------------------ display ----------------------------------
void drawScreen() {
  const char *st = effectiveState();
  unsigned long ageS = haveLine ? (millis() - lastLineMs) / 1000UL : 999UL;

  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);

  display.setTextSize(1);
  display.setCursor(0, 0);
  display.print("RAG heartbeat");
  if (clientConnected) {
    display.setCursor(112, 0);
    display.print("*");
  }

  display.setTextSize(2);
  display.setCursor(0, 12);
  display.print(st);

  display.setTextSize(1);
  display.setCursor(0, 34);
  if (strcmp(st, "WAIT") == 0) {
    display.print("waiting for host");
  } else if (strcmp(st, "LOST") == 0) {
    display.print("no data from PC");
  } else {
    char buf[24];
    snprintf(buf, sizeof(buf), "lat %d ms", latencyMs);
    display.print(buf);
  }

  display.setCursor(0, 44);
  if (haveLine) {
    char buf[40];
    snprintf(buf, sizeof(buf), "q%d/m idx%d %s", qpm, chunks, mock ? "MOCK" : "real");
    display.print(buf);
  }

  display.setCursor(0, 54);
  char foot[28];
  if (haveLine) {
    snprintf(foot, sizeof(foot), "beat %lus ago %s", ageS, ((millis() / 500) % 2) ? "." : " ");
  } else {
    snprintf(foot, sizeof(foot), "no beat yet %s", ((millis() / 500) % 2) ? "." : " ");
  }
  display.print(foot);

  if (message.length() > 0 && strcmp(st, "OK") != 0) {
    display.setCursor(0, 26);
    display.print(message.substring(0, 21));
  }

  display.display();
}

// ------------------------------- setup -----------------------------------
void setup() {
  pinMode(BUZZER_PIN, OUTPUT);
  buzzerOff();

  Wire.begin(I2C_SDA, I2C_SCL);
  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    // No OLED: still beep so the board is not silently dead.
    for (int i = 0; i < 6; i++) {
      buzzerOn();
      delay(120);
      buzzerOff();
      delay(120);
    }
  }
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.print("RAG heartbeat");
  display.setCursor(0, 16);
  display.print("starting BLE...");
  display.display();

  BLEDevice::init(BLE_NAME);
  server = BLEDevice::createServer();
  server->setCallbacks(new ServerCallbacks());

  BLEService *svc = server->createService(NUS_SERVICE_UUID);
  BLECharacteristic *rx = svc->createCharacteristic(
      NUS_RX_UUID, BLECharacteristic::PROPERTY_WRITE | BLECharacteristic::PROPERTY_WRITE_NR);
  rx->setCallbacks(new RxCallbacks());

  txChar = svc->createCharacteristic(NUS_TX_UUID, BLECharacteristic::PROPERTY_NOTIFY);
  txChar->addDescriptor(new BLE2902());

  svc->start();

  BLEAdvertising *adv = BLEDevice::getAdvertising();
  adv->addServiceUUID(NUS_SERVICE_UUID);
  adv->setScanResponse(true);
  adv->setMinPreferred(0x06);
  BLEDevice::startAdvertising();

  display.setCursor(0, 32);
  display.print("advertising as " BLE_NAME);
  display.display();
  beepPattern(1, 80, 0);
}

// -------------------------------- loop -----------------------------------
unsigned long lastDraw = 0;
const char *lastState = "WAIT";

void loop() {
  serviceBuzzer();

  // Transitions that happen without a new packet (LOST, and recovery).
  const char *cur = effectiveState();
  if (strcmp(cur, lastState) != 0) {
    if (isBadState(cur) || strcmp(cur, "OK") == 0) {
      applyState(cur, lastState);
    }
    lastState = cur;
  }

  static unsigned long lastRemind = 0;
  if (isBadState(cur)) {
    if (millis() - lastRemind > REMIND_MS) {
      lastRemind = millis();
      beepPattern(1, 150, 0);
    }
  } else {
    lastRemind = millis();
  }

  if (millis() - lastDraw > 300) {
    lastDraw = millis();
    drawScreen();
  }
  delay(10);
}
