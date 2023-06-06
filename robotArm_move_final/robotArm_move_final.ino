#include <HCPCA9685.h>
#define I2CAdd 0x40
HCPCA9685 HCPCA9685(I2CAdd);
String str = "";
int angle[3] = {90, 45, 135};

void angle_Set(String nstr)//get angle order from pyhton and save in angle[4]-> when this method activate arr angle is changed
{
  int temp = 0;
  int index = nstr.indexOf(' ');
  for(int i = 0; i < 3; i++){
    angle[i] = nstr.substring(temp, index).toInt();
    temp = index + 1;
    index = nstr.indexOf(' ', temp);
  }
}


void setup() {
  HCPCA9685.Init(SERVO_MODE);
  HCPCA9685.Sleep(false);
  Serial.begin(9600);
  Serial.setTimeout(10);
  for(int i=0; i<3;i++){
    angle[i]=map(angle[i],0,180,0,400);
    HCPCA9685.Servo(i+4, angle[i]);
  }
}


void loop() {
  if(Serial.available()){
    str = Serial.readString();
    angle_Set(str);
    for(int i = 0; i < 3; i++){
      HCPCA9685.Servo(i+4, angle[i]);
    }
    delay(20);
    Serial.println("Done");
    
  }
}
