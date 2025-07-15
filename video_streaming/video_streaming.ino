//通过内置的web streaming功能连接热点然后进行video streaming
#include "esp_camera.h"
#include <WiFi.h>

#define CAMERA_MODEL_XIAO_ESP32S3

#include "camera_pins.h"

//设置WIFI连接

const char* ssid="MCTWOATCWB";
const char* password="43u)F030";

void m_startCameraServer();
void setupLedFlash(int pin);
void setup() {
  Serial.begin(115200);
  while(!Serial);
  Serial.setDebugOutput(true);
  Serial.println();

  camera_config_t config;
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

  config.pin_xclk = XCLK_GPIO_NUM;//外部时钟引脚，提供时钟信号
  config.pin_pclk = PCLK_GPIO_NUM;//像素时钟引脚
  config.pin_vsync = VSYNC_GPIO_NUM;//垂直同步引脚
  config.pin_href = HREF_GPIO_NUM;//行同步引脚
  config.pin_sscb_sda = SIOD_GPIO_NUM;//SCCB数据引脚，用户摄像头进行数据通信
  config.pin_sscb_scl = SIOC_GPIO_NUM;//SCCB时钟引脚
  config.pin_pwdn = PWDN_GPIO_NUM;//电源控制引脚
  config.pin_reset = RESET_GPIO_NUM;//复位引脚
  config.xclk_freq_hz = 20000000;//外部时钟频率
  config.frame_size = FRAMESIZE_UXGA;//图像帧大小//ultra extended graphics array 1600x1200
  config.pixel_format = PIXFORMAT_JPEG; // for streaming
  //config.pixel_format = PIXFORMAT_RGB565; // for face detection/recognition
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;//图像抓取策略
  config.fb_location = CAMERA_FB_IN_PSRAM;//帧缓冲区的位置和状态
  config.jpeg_quality = 12;//JPEG编码质量
  config.fb_count = 1;//帧缓冲区的数量

  if(config.pixel_format==PIXFORMAT_JPEG){
    if(psramFound()){//检查是否找到了PSRAM（外部静态随即处理存储器）
      config.jpeg_quality=10;
      config.fb_count=2;
      config.grab_mode=CAMERA_GRAB_LATEST;//始终获取最新的
    }
    else{
      config.frame_size=FRAMESIZE_SVGA;//super VGA 800x600
      config.fb_location= CAMERA_FB_IN_DRAM;
    }
  }
  else{
    config.frame_size=FRAMESIZE_240X240;
      #if CONFIG_IDF_TARGET_ESP32S3
      config.fb_count=2;
      #endif
  }
  #if defined(CAMERA_MODEL_ESP_EYE)
    pinMode(13,INPUT_PULLUP);
    pinMode(13,INPUT_PULLUP);
  #endif
  
  esp_err_t err=esp_camera_init(&config);
  if(err!=ESP_OK){
    Serial.printf("Camera init failed with error 0x%x",err);
    return;
  }

  //用于描述和管理摄像头传感器的属性和功能
  sensor_t *s=esp_camera_sensor_get();//获取与摄像头传感器相关的信息或配置
  //返回一个指向sensor_t结构的指针
  //设置OV3660_PID相机参数
  if(s->id.PID==OV3660_PID){
    s->set_vflip(s,1);//开启垂直翻转
    s->set_brightness(s,1);//增加亮度
    s->set_saturation(s,-2);//降低饱和度
  }
  if(config.pixel_format=PIXFORMAT_JPEG){
    s->set_framesize(s,FRAMESIZE_QVGA);//quarter VGA 320x240
  }
  //其他类型的相机模块
  // #if defined(CAMERA_MODEL_M5STACK_WIDE) || defined(CAMERA_MODEL_M5STACK_ESP32CAM)
  // s->set_vflip(s, 1);
  // s->set_hmirror(s, 1);
  // #endif

  // #if defined(CAMERA_MODEL_ESP32S3_EYE)
  //   s->set_vflip(s, 1);
  // #endif
  #if defined(LED_GPIO_NUM)
    setupLedFlash(LED_GPIO_NUM);
  #endif
  WiFi.begin(ssid,password);
  WiFi.setSleep(false);
  while(WiFi.status()!=WL_CONNECTED){
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("WiFi connected");
  m_startCameraServer();

  Serial.print("Camera Ready! Use 'http://");
  Serial.print(WiFi.localIP());
  Serial.println("' to connect");


}

void loop() {
    // 主循环代码
    delay(10000);
}