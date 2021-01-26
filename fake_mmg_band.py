import json
import paho.mqtt.client as mqtt
import time
import numpy as np

FILE = 'data/gestures/idle_0.npy'
LABEL = FILE.split('/')[2][:-4]
gesture_data = np.load(FILE).tolist()

GESTURES = [
    'idle',                 # 0 -> 0
    'fist',                 # 1 -> 1
    'flexion',              # 2 -> 2
    'extension',            # 3 -> 3
    'pinch thumb-index',    # 4 -> 6
    'pinch thumb-middle',   # 5 -> 7
    'pinch thumb-ring',     # 6 -> 8
    'pinch thumb-small'     # 7 -> 9
] 

print("Gesture data: {}.\nSize:".format(LABEL), len(gesture_data))

BROKER_IP = "192.168.1.26"
BROKER_PORT= 1883
DATA_STREAM_TOPIC="sensors/data/mmg"
CONTROL_TOPIC = "sensors/control/mmg"

# ================== MAIN ================== #

def connect_to_broker(client:mqtt.Client):
    client.connect(BROKER_IP, BROKER_PORT)
    print("Connected with broker: {}:{}".format(BROKER_IP, str(BROKER_PORT)))

def fake_mmg():
    mqtt_client = mqtt.Client()
    mqtt_client.on_connect = lambda c, userdata, flags, rc: c.subscribe(DATA_STREAM_TOPIC)
    connect_to_broker(mqtt_client)
    
    print("Ready to send...")
    timer = 0
    for i in range(len(gesture_data)):

        time.sleep(0.02) # SEND DATA EVERY 20 ms

        cutted_array = gesture_data[i: i+4]
        dict_to_json = {
            "data": cutted_array,
            "packets": len(cutted_array),
            "timestamp": timer,
            "freq": 50,
            "channels": 8
        }
        timer += 20000
        i+=1
        print(json.dumps(dict_to_json))
        print("Data from gesture:", )
        mqtt_client.publish(DATA_STREAM_TOPIC, json.dumps(dict_to_json))

if __name__ == "__main__":
    fake_mmg()
