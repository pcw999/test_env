import cv2
import numpy as np
import socket

# Create a UDP socket and bind it to a local address and port
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket.bind(('0.0.0.0', 5000))

# Set the address and port of the remote peer
remote_address = ('192.168.0.30', 5000)

# Set the maximum size of a UDP packet
max_packet_size = 61440

# Set a timeout on the socket receive operation to avoid blocking for too long
udp_socket.settimeout(0.01)

# Set the socket to non-blocking mode
udp_socket.setblocking(False)

# Start capturing frames from the webcam
cap = cv2.VideoCapture(0)

# Calculate the number of packets required to send a video frame
frame_size = 2764800  # 1280x720 with 3 color channels
packet_count = (frame_size + max_packet_size - 1) // max_packet_size

# Main loop for sending and receiving video frames
while True:
    # Capture a video frame from the webcam and resize it to 1280x720
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))

    # Convert the video frame to a byte string and split it into packets
    send_data = frame.tobytes()
    packets = [send_data[i:i+max_packet_size] for i in range(0, len(send_data), max_packet_size)]

    # Send each packet sequentially to the remote peer
    for i in range(packet_count):
        while True:
            try:
                udp_socket.sendto(packets[i], remote_address)
                break
            except BlockingIOError:
                pass

    try:
        # Receive packets from the remote peer and reassemble them into a video frame
        received_packets = []
        while len(received_packets) < packet_count:
            packet, _ = udp_socket.recvfrom(max_packet_size)
            received_packets.append(packet)
        received_data = b''.join(received_packets)
        received_frame = np.frombuffer(received_data, dtype=np.uint8)
        received_frame = received_frame.reshape(720, 1280, 3)

        # Display the received video frame on the screen
        cv2.imshow("Received Frame", received_frame)
    except socket.timeout:
        # If no data is received, do nothing and continue the loop
        pass

    # Check for user input to quit the program
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and close the socket
cap.release()
cv2.destroyAllWindows()
udp_socket.close()
