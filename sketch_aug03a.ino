int analogPin1 = 1;
int analogPin2 = 2;
int analogPin3 = 3;
int analogPin4 = 4;
int analogPin5 = 5;

// outside leads to ground and +5V

int val1 = 0;
int val2 = 0;
int val3 = 0;
int val4 = 0;
int val5 = 0;
#include <Arduino.h>
void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:


  int times = 0;
  digitalWrite(LED_BUILTIN, HIGH);   // turn the LED on (HIGH is the voltage level)
  delay(150);    // wait for 150 ms
  val1 = analogRead(analogPin1);    // read the input pin
  val2 = analogRead(analogPin2);
  val3 = analogRead(analogPin3);
  val4 = analogRead(analogPin4);
  val5 = analogRead(analogPin5);
  Serial.print(val1);// debug value
  Serial.print(",");
  Serial.print(val2);
  Serial.print(",");
  Serial.print(val3);
  Serial.print(",");
  Serial.print(val4);
  Serial.print(",");
  Serial.print(val5);
  delay(150);
  digitalWrite(LED_BUILTIN, LOW);    // turn the LED off by making the voltage LOW
  delay(20);
  for (int i = 0; i < 9; ++i) {
    val1 = analogRead(analogPin1);// read the input pin
    val2 = analogRead(analogPin2);
    val3 = analogRead(analogPin3);
    val4 = analogRead(analogPin4);
    val5 = analogRead(analogPin5);
    Serial.print(",");
    Serial.print(val1);// debug value
    Serial.print(",");
    Serial.print(val2);
    Serial.print(",");
    Serial.print(val3);
    Serial.print(",");
    Serial.print(val4);
    Serial.print(",");
    Serial.print(val5);
    
    delay(75);
    times = times + 1;
  }
  Serial.println("");
  delay(3000);

}
