import cv2
import numpy as np
import socket
import struct

socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
socket.bind(('0.0.0.0', 5000))

opp_addr = ('192.168.0.30', 5000)
socket.timeout(0.01)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    data = frame.tobytes()
    socket.sendto(data, opp_addr)

    try:
        socket.recvfrom(2764800)
        opp_frame = np.frombuffer(data, dtype=np.uint8)
        opp_frame = opp_frame.reshape(720, 1280, 3)
    except:
        pass

    cv2.imshow("Received Frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
socket.close()