import socket
import csv
import os
import time
import argparse
import sys


parser = argparse.ArgumentParser()
parser.add_argument("-r", "--RU", help="specify if this is the RU", action="store_true")
parser.add_argument("-i", "--ip", help="enter the distant end IP address", type=str, required=True)
parser.add_argument("-f", "--file", help="full path to the csv file", type=str, required=True)
args = parser.parse_args()

# These are the default values
RU_PORT = 5005
DU_PORT = 5115
data_size = 0

# add some arguments so we can specify a few options at run time
DU = not args.RU
file_name = args.file

if DU:
    local_port = DU_PORT
    distant_port = RU_PORT
else:
    local_port = RU_PORT
    distant_port = DU_PORT

Distant_IP = args.ip

print("UDP target IP: %s" % Distant_IP)

# sending UDP socket
send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# receiving UDP socket
rec_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
rec_sock.bind(('', local_port))

with open(file_name, 'r') as csvfile:
    datareader = csv.reader(csvfile)
    row1 = next(datareader)

    if DU:  # The DU should always start
        print("I am the DU, I start communication")
        row2 = next(datareader)
        # print(row2)
        if row2[1] != 'b8:ce:f6:5e:6b:2a':
        # if row2[2] != '192.168.1.9':
            print('RU starts, send start message')
            send_sock.sendto(str.encode('Start'), (Distant_IP, distant_port))
            start_time = time.time()
            print('starting experiment')
            #print('listening for packet number %s' % row2[0])
            # we also need to wait and listen for the first message
            while time.time() - start_time <= float(row2[0]):
                continue

        else:
            # it is our turn to start
            data_size = int(row2[3]) -42 
            print('DU starting')
            start_time = time.time()
            Sdata = os.urandom(data_size)
            send_sock.sendto(Sdata, (Distant_IP, distant_port))

        for row in datareader:
            if row[1] == 'b8:ce:f6:5e:6b:2a':
            # if row[2] == '192.168.1.9':
                data_size = int(row[3]) -42 
                Sdata = os.urandom(data_size)
                while time.time()-start_time <= float(row[0]):  # but first, we have to check the time!
                    continue
                send_sock.sendto(Sdata, (Distant_IP, distant_port))
                print('Sending packet number %s' % row[0])

            else:
                # we should listen until we get data, or it is our turn to send again
                print('listening for packet number %s' % row[0])
                while time.time() - start_time <= float(row[0]):
                    continue

    else:  # if we are the RU, we need to wait for a message from the DU before moving on
        print("waiting for DU")

        while True:
            data, address = rec_sock.recvfrom(8192)
            if data:
                start_time = time.time()
                print("Starting experiment")
                break

        for row in datareader:
            if row[1] == '6c:ad:ad:00:04:de':
            # if row[2] == '192.168.1.8':
                data_size = int(row[3]) -42 
                Sdata = os.urandom(data_size)
                while time.time()-start_time <= float(row[0]):  # but first, we have to check the time!
                    continue
                send_sock.sendto(Sdata, (Distant_IP, distant_port))
                print('Sending packet number %s' % row[0])

            else:
                print('listening for packet number %s' % row[0])
                # we should listen until we get data, or it is our turn to send again
                while time.time() - start_time <= float(row[0]):
                    continue

print('Test completed with the following parameters:')
print("UDP target IP: %s" % Distant_IP)
print("File used: %s" % file_name)
