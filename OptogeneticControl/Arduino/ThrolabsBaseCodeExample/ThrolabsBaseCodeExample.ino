

int analogPin = A3; // potentiometer wiper (middle terminal) connected to analog pin 3
                    // outside leads to ground and +5V


int val = 0;  // variable to store the value read

void setup() {
    // put your setup code here, to run once:
  Serial.begin(9600);           //  setup serial
  pinMode(A0, OUTPUT);
}

void loop() {
   // put your main code here, to run repeatedly:
  val = analogRead(analogPin);  // read the input pin
  if(val > 0){
    digitalWrite(A0, HIGH);
  } else {
    digitalWrite(A0, LOW);
  }
  Serial.println(val);          // debug value
}
