#!/bin/bash

ip=10.48.141.212

ssh pi@$ip 'sudo python3 /home/pi/Project/raspi_main.py --mode dev' &