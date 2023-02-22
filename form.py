import socket
import cv2
from cvzone.HandTrackingModule import HandDetector

# socket
UDP_IP = 'opponent_ip'
UDP_PORT = 'opponent_port'
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', 5000))
sock.settimeout(0.01)

# handtracking
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8, maxHands=1)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=False)

    if hands:
        lmList = hands[0]['lmList']
        cv2.circle(img, (lmList[8][0], lmList[8][1]), 10, (255, 0, 255), cv2.FILLED)
        pointIndex = str(lmList[8][0]) + '/' + str(lmList[8][1])
        sock.sendto(pointIndex.encode(), (UDP_IP, UDP_PORT))

    try:
        data, _ = sock.recvfrom(100)
        pointIndex = data.decode()
        x, y = pointIndex.split('/')
        if len(data) != 0 :
            cv2.circle(img, (int(x), int(y)), 10, (0, 255, 255), cv2.FILLED)
    except socket.timeout:
        pass
  
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break

sock.close()
cv2.destroyAllWindows()