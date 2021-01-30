import json
import paho.mqtt.client as mqtt
import time
import numpy as np
import sys
import getopt

def get_args(argv):
    gesture = 0
    try:
        opts, args = getopt.getopt(argv, "g:", ["gesture="])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-g':
            gesture = arg
        else: sys.exit(2)

    return int(gesture)

FILES = [
    'idle_0.npy',
    'fist_1.npy',
    'flexion_2.npy',
    'extension_3.npy',
    'pinch_thumb-index_4.npy',
    'pinch_thumb-middle_5.npy',
    'pinch_thumb-ring_6.npy',
    'pinch_thumb-small_7.npy',
]

GESTURES = [
    'idle',                 # 0 -> 0
    'fist',                 # 1 -> 1
    'flexion',              # 2 -> 2
    'extension',            # 3 -> 3
    'pinch_thumb-index',    # 4 -> 6
    'pinch_thumb-middle',   # 5 -> 7
    'pinch_thumb-ring',     # 6 -> 8
    'pinch_thumb-small'     # 7 -> 9
] 

BROKER_IP = "192.168.1.26"
BROKER_PORT= 1883
DATA_STREAM_TOPIC="sensors/data/mmg"
CONTROL_TOPIC = "sensors/control/mmg"

# ================== MAIN ================== #

def connect_to_broker(client:mqtt.Client):
    client.connect(BROKER_IP, BROKER_PORT)
    print("Connected with broker: {}:{}".format(BROKER_IP, str(BROKER_PORT)))

def fake_mmg(gesture):
    gesture_data = np.load(f"data/gestures/{FILES[gesture]}").tolist()
    print(f"Gesture: {GESTURES[gesture]}\nLabel: \'{gesture}\'.\nAvailable data:".format(gesture), len(gesture_data))

    mqtt_client = mqtt.Client()
    mqtt_client.on_connect = lambda c, userdata, flags, rc: c.subscribe(DATA_STREAM_TOPIC)
    connect_to_broker(mqtt_client)
    
    input("Ready to send. \n\nTo start transmission, press ENTER!")
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
        i+=4
        print(f"\nGesture: {GESTURES[gesture]}\nLabel: \'{gesture}\'\nData: {i}/{len(gesture_data)}\nFake time: {timer}")
        mqtt_client.publish(DATA_STREAM_TOPIC, json.dumps(dict_to_json))

if __name__ == "__main__":
    gesture = get_args(sys.argv[1:])
    fake_mmg(gesture)
