from scapy.all import PcapReader, IP, Ether
import csv
import argparse
import fnmatch
import os



ptp_message_types = {0: "Sync", 
                     1: "Delay_Req",
                     8: "Follow_Up", 
                     9: "Delay_Resp",
                     11: "Announce"}


def pcap_to_csv(input_file, output_file, chunk_size=1000):
    # Open the pcap file for reading
    with PcapReader(input_file) as pcap_reader:
        # Open the CSV file for writing
        start_time = None
        with open(output_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)

            # Write header row
            csv_writer.writerow(['Time', 'Source', 'Destination', 'Protocol', 'Length', 'SequenceID', 'MessageType'])
            
            # Process packets in chunks
            count = 0
            chunk = []
            for packet in pcap_reader:
                count+=1
                
                if start_time is None:
                    start_time = packet.time

                if Ether in packet:
                    
                    src_eth = packet[Ether].src
                    dst_eth = packet[Ether].dst
                    
                    if src_eth and dst_eth:
                        packet_length = len(packet)
                        if packet[Ether].type == 35063:
                            protocol = "PTPv2"
                            sequenceid = int.from_bytes(packet.load[30:32], byteorder='big')
                            messagetype = int.from_bytes(packet.load[:1], byteorder='big')
                            time= packet.time-start_time
                            source =src_eth
                            destination = dst_eth
                            length = packet_length
                            # Add packet information to the current chunk
                            chunk.append([time, source, destination, protocol, length, sequenceid, messagetype])
    
                        else:
                            protocol = '-'
                            sequenceid = '-'
                            messagetype = '-'
                            time= packet.time-start_time
                            source =src_eth
                            destination = dst_eth
                            length = packet_length
                            # Add packet information to the current chunk
                            chunk.append([time, source, destination, protocol, length, sequenceid, messagetype])
                                
                    # If chunk size is reached, write to CSV and reset chunk
                if count % chunk_size == 0:
                    csv_writer.writerows(chunk)
                    chunk = []
                    print(f"Processed {count} packets")
            
            # Write any remaining packets to CSV
            if chunk:
                csv_writer.writerows(chunk)
                print(f"Processed {count} packets")

# Example usage
if __name__ == '__main__':
    # Load the PCAP file

    #pcap_to_csv(args.input, args.output, args.chunk)
    recursion = True
    print('STARTING')
    for root, dirnames, filenames in os.walk('./'):
        for filename in fnmatch.filter(filenames, '*.pcap'):
            path = os.path.join(root, filename)
            path = os.path.splitext(path)[0]
            print(path)
    for root, dirnames, filenames in os.walk('./'):
        for filename in fnmatch.filter(filenames, '*.pcap'):
            path = os.path.join(root, filename)
            path = os.path.splitext(path)[0]
            print('TO PROCESS '+path)

            if not os.path.exists(path+'.csv'):
                recursion = recursion and False
                pcap_to_csv(path+'.pcap', path+'.csv', 10000)
                os.system('y | rm '+path+'.pcap')
                print("CSV CREATED FOR "+path)
            else:
                print("CSV EXISTS FOR "+path)

        if recursion == False:
            os.system('python3 pcap_csv_converter.py')


