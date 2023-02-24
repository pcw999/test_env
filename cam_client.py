import socket
import cv2

# socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    d = img.flatten()
    s = d.tostring()
  
    for i in range(45):
        sock.sendto(bytes([i]) + s[i*61440:(i+1)*61440], ('192.168.0.30', 5000))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

sock.close()