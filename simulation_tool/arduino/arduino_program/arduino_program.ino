#define SERIAL_SPEED 115200
#define INPUT_PORT 12
#define OUTPUT_PORT 11

unsigned long timeout = 30000; // 30 seconds
unsigned long duration;
unsigned long t1,t2;
double penalty_th, delay_time;

int state = 0;
int prev_state = 0;

bool debug = 0 ;

void setup() {
    Serial.begin(SERIAL_SPEED);   // Open serial communications and wait for port to open:
    while (!Serial) {};           // wait for serial port to connect. Needed for native USB port only
    pinMode(OUTPUT_PORT, OUTPUT);
    pinMode(INPUT_PORT, INPUT);
    digitalWrite(OUTPUT_PORT, LOW);
    Serial.println(0xfffff,HEX);
}

float wait_for_high_state(void){
    digitalWrite(OUTPUT_PORT, HIGH);
    t1 = millis();
    delay(10);
    while(digitalRead(INPUT_PORT) != 0){
        if(millis()-t1 > timeout) return -1.0;
    }
    t2 = millis();
    while(digitalRead(INPUT_PORT) == 0){
        if(millis()-t1 > timeout) return -1.0;
    }
//    while(digitalRead(INPUT_PORT) != 0){
//        if(millis()-t1 > timeout) return -1.0;
//    }    
    return (float)(t2-t1)/1000.0;

    t1 = millis();
    prev_state = digitalRead(INPUT_PORT); 
}



void loop() {
//  while(1){
//    state = digitalRead(INPUT_PORT); 
//    if (prev_state != state){
//      t2 = millis();
//      duration = (float)(t2-t1)/1000.0;
//      Serial.print("state=");
//      Serial.print(state);
//      Serial.print(";time=");
//      Serial.println(duration);
//      t1 = millis();
//      prev_state = digitalRead(INPUT_PORT);
//       
//    }
//    delay(100);
//  }
//  
//  if (Serial.available()) {
//            penalty_th = Serial.parseFloat();
//            if (penalty_th> 0){
//                digitalWrite(OUTPUT_PORT, HIGH); 
//                Serial.println("set_high");
//            }else{
//                digitalWrite(OUTPUT_PORT, LOW); 
//                Serial.println("set_low");
//            }
//            while(Serial.available()){
//            Serial.read();
//        }
//  }

//  
    if (Serial.available()) {
        penalty_th = Serial.parseFloat();
        if (debug){
          Serial.print("penalty_th=");
          Serial.println(penalty_th);
        }
        delay_time = wait_for_high_state();
        if (delay_time > penalty_th){
          delay(1000);
           if (debug) {
            Serial.println("penalty");
           }
            delay(1000);
          }
        
        digitalWrite(OUTPUT_PORT, LOW);
        Serial.println(delay_time,2);
        while(Serial.available()){
            Serial.read();
        }
  }
}
        
  
