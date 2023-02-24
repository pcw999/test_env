import socket
import cv2
import numpy
import threading

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
PACKET_SIZE = 8192

def send_packets(sock, address, queue):
    while True:
        frame = queue.get()
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result, imgencode = cv2.imencode('.jpg', frame, encode_param)
        data = numpy.array(imgencode)
        stringData = data.tostring()

        for i in range(0, len(stringData), PACKET_SIZE):
            packet = bytes([i // PACKET_SIZE]) + stringData[i:i+PACKET_SIZE]
            sock.sendto(packet, address)

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    cap = cv2.VideoCapture(0)

    address = ('192.168.0.30', 5000)

    queue = queue.Queue()
    sender_thread = threading.Thread(target=send_packets, args=(sock, address, queue))
    sender_thread.daemon = True
    sender_thread.start()

    while True:
        ret, frame = cap.read()
        if ret:
            queue.put(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    sock.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
