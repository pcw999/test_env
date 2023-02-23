import cv2
import numpy as np
import socket
import struct

udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket.bind(('0.0.0.0', 5000))

opp_addr = ('192.168.0.30', 5000)
udp_socket.settimeout(0.01)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    frame_bytes = frame.tobytes()

    # Split the frame into packets of size 65535 bytes or less
    packets = [frame_bytes[i:i+65535] for i in range(0, len(frame_bytes), 65535)]

    for packet in packets:
        udp_socket.sendto(packet, opp_addr)

    try:
        # Receive the acknowledgment packet from the receiver
        ack, _ = udp_socket.recvfrom(4)
        if ack == b'ACK\n':
            opp_frame = np.frombuffer(frame_bytes, dtype=np.uint8)
            opp_frame = opp_frame.reshape(720, 1280, 3)
            cv2.imshow("Received Frame", opp_frame)
    except:
        pass

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
udp_socket.close()
