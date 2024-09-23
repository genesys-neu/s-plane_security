from scapy.all import *
from scapy.all import Ether

class SyncPTP:
    def __init__(self,original_sync):
        
        self.eth_layer = Ether(
            src = original_sync[Ether].src,
            dst = original_sync[Ether].dst,
            type = original_sync[Ether].type
        )
        self.ptp_layer = original_sync.load

        self.packet = self.eth_layer / self.ptp_layer

    def new_Ether_src(self, new_value):
        self.eth_layer.src = new_value
        self.packet = self.eth_layer / self.ptp_layer

    def new_Ether_dst(self, new_value):
        self.eth_layer.dst = new_value
        self.packet = self.eth_layer / self.ptp_layer

    def new_majorSdoId_messageType(self, offset):
        new_value = offset.to_bytes((offset.bit_length() + 7) // 8, byteorder='big')
        self.ptp_layer = new_value + self.ptp_layer[1:]
        self.packet = self.eth_layer / self.ptp_layer

    def new_minorVersionPTP_versionPTP(self, offset):
        new_value = offset.to_bytes((offset.bit_length() + 7) // 8, byteorder='big')
        self.ptp_layer = self.ptp_layer[:1]+new_value+self.ptp_layer[2:]
        self.packet = self.eth_layer / self.ptp_layer
    
    def new_messageLength(self, offset):
        new_value = offset.to_bytes((offset.bit_length() + 7) // 8, byteorder='big')
        self.ptp_layer = self.ptp_layer[:2]+new_value+self.ptp_layer[4:]
        self.packet = self.eth_layer / self.ptp_layer

    def new_domainNumber(self, offset):
        new_value = offset.to_bytes((offset.bit_length() + 7) // 8, byteorder='big')
        self.ptp_layer = self.ptp_layer[:4]+new_value+self.ptp_layer[5:]
        self.packet = self.eth_layer / self.ptp_layer

    def new_minorSdoId(self, offset):
        new_value = offset.to_bytes((offset.bit_length() + 7) // 8, byteorder='big')
        self.ptp_layer = self.ptp_layer[:5]+new_value+self.ptp_layer[6:]
        self.packet = self.eth_layer / self.ptp_layer

    def new_flags(self, offset):
        new_value = offset.to_bytes((offset.bit_length() + 7) // 8, byteorder='big')
        self.ptp_layer = self.ptp_layer[:6]+new_value+self.ptp_layer[8:]
        self.packet = self.eth_layer / self.ptp_layer

    def new_correctionField(self, offset):
        new_value = offset.to_bytes((offset.bit_length() + 7) // 8, byteorder='big')
        self.ptp_layer = self.ptp_layer[:8]+new_value+self.ptp_layer[16:]
        self.packet = self.eth_layer / self.ptp_layer

    def new_messageTypeSpecific(self, offset):
        new_value = offset.to_bytes((offset.bit_length() + 7) // 8, byteorder='big')
        self.ptp_layer = self.ptp_layer[:16]+new_value+self.ptp_layer[20:]
        self.packet = self.eth_layer / self.ptp_layer

    def new_ClockIdentity(self,new_value):
        self.ptp_layer = self.ptp_layer[:20]+ new_value + self.ptp_layer[28:]
        self.packet = self.eth_layer / self.ptp_layer

    def new_SourcePortID(self,offset):
        new_value = offset.to_bytes(2, byteorder='big')
        self.ptp_layer = self.ptp_layer[:28]+ new_value + self.ptp_layer[30:]
        self.packet = self.eth_layer / self.ptp_layer

    def new_SequenceID(self,offset):
        new_value = offset.to_bytes((offset.bit_length() + 7) // 8, byteorder='big')
        new_value = offset.to_bytes((offset.bit_length() + 7) // 8, byteorder='big')
        self.ptp_layer = self.ptp_layer[:30]+ new_value + self.ptp_layer[32:]
        self.packet = self.eth_layer / self.ptp_layer

    def Increment_SequenceID(self,offset):
        decoded_int = int.from_bytes(self.ptp_layer[30:32], byteorder='big')
        decoded_int+=offset
        new_value = decoded_int.to_bytes((decoded_int.bit_length() + 7) // 8, byteorder='big')
        self.ptp_layer = self.ptp_layer[:30]+ new_value + self.ptp_layer[32:]
        self.packet = self.eth_layer / self.ptp_layer

    def new_controlField(self,offset):
        new_value = offset.to_bytes((offset.bit_length() + 7) // 8, byteorder='big')
        self.ptp_layer = self.ptp_layer[:32]+ new_value + self.ptp_layer[33:]
        self.packet = self.eth_layer / self.ptp_layer
    
    def new_logMessagePeriod(self,offset):
        new_value = offset.to_bytes((offset.bit_length() + 7) // 8, byteorder='big')
        self.ptp_layer = self.ptp_layer[:33]+ new_value + self.ptp_layer[34:]
        self.packet = self.eth_layer / self.ptp_layer
    
    def new_originTimestampSeconds(self, decoded_int):
        new_value = decoded_int.to_bytes(6, byteorder='big')
        self.ptp_layer = self.ptp_layer[:34] + new_value + self.ptp_layer[40:]
        self.packet = self.eth_layer / Raw(load=self.ptp_layer)

    def new_originTimestampNanoseconds(self, decoded_int):
        new_value = decoded_int.to_bytes(4, byteorder='big')
        self.ptp_layer = self.ptp_layer[:40] + new_value + self.ptp_layer[44:]
        self.packet = self.eth_layer / Raw(load=self.ptp_layer)

    def new_padding(self,offset):
        new_value = offset.to_bytes((offset.bit_length() + 7) // 8, byteorder='big')
        self.ptp_layer = self.ptp_layer[:44]+ new_value
        self.packet = self.eth_layer / self.ptp_layer
