
import socket
import numpy
import cv2

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', 5000))

s = [b'\xff' * 61440 for x in range(45)]

while True:
    picture = b''

    data, addr = sock.recvfrom(61441)
    s[data[0]] = data[1:61441]

    if data[0] == 44:
        for i in range(45):
            picture += s[i]

        frame = numpy.fromstring(picture, dtype=numpy.uint8)
        frame = frame.reshape(720, 1280, 3)
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

sock.close()