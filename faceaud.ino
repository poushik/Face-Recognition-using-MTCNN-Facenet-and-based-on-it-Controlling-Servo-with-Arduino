#include <Servo.h>
Servo motor; 
int data;
void setup() {
  Serial.begin(9600);
  motor.attach(10);
}

void loop() {
  while(Serial.available()){
    data=Serial.read();
    if(data=='1'){
      motor.write(90);
      delay(1000);
    }
    else{
      motor.write(0);
    }
  }
}
