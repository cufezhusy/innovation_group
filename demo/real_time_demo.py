import threading
import time
import msvcrt
from tick_price import *
from helper import *
import sys
import random
import keras


tick_interval = 0.02
pre_heat_time = 100
random_trade_prob  = 0.02


mpes_obj = market_price_evaluator_service(file="tick_data.txt", sym="EUR=")
mpes_obj.pre_load()

model = keras.models.load_model("final_model.h5")

input("Real time demo is ready, any key to start")

for s in range(len(mpes_obj.temp_list)):
    time.sleep(tick_interval)
    print(mpes_obj.get_tick(s)[1])

    if s > pre_heat_time and random.uniform(0,1) < random_trade_prob:
        print("---------------------------------------------")
        price = input("New trade come in, please input the price:")
        x = generate_x(price,mpes_obj,s)
        y_hat = model.predict(x)
        predict = 1 if y_hat[0, 1] > 0.5 else 0
        output_str = "Abnormal trade " if y_hat[0, 1] >= 0.5 else "Normal trade"
        print("Model Says : %s " % output_str)
        input("Press any key to continue")
        print("---------------------------------------------")