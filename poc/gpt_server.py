import socket
import cv2
import threading
import queue
import numpy

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
PACKET_SIZE = 8192

def receive_packets(sock, frame_queue):
    while True:
        data, addr = sock.recvfrom(PACKET_SIZE)
        frame_queue.put(data)

def process_frames(frame_queue):
    s = [b'' for x in range(45)]
    while True:
        picture = b''
        for i in range(45):
            data = frame_queue.get()
            s[data[0]] = data[1:]
            if i == 44:
                for j in range(45):
                    picture += s[j]

        if picture:
            frame = cv2.imdecode(numpy.frombuffer(picture, dtype=numpy.uint8), cv2.IMREAD_COLOR)
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', 5000))

    frame_queue = queue.Queue()
    receiver_thread = threading.Thread(target=receive_packets, args=(sock, frame_queue))
    receiver_thread.daemon = True
    receiver_thread.start()

    process_frames(frame_queue)

    sock.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
