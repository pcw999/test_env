import json
import datetime
import time
import math
import random
import cvzone
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, Response, request, redirect, url_for, session
from flask_socketio import SocketIO, emit, join_room
import os
import ssl
from distutils.util import strtobool
import aiohttp
from aiohttp import web
import jinja2
import aiohttp_jinja2
import uuid
from engineio.payload import Payload
import socket

Payload.max_decode_packets = 200

app = Flask(__name__)
app.config['SECRET_KEY'] = "roomfitisdead"

socketio = SocketIO(app, cors_allowed_origins='*')

### 임시 변수 => snake 게임 데이터를 emit 할지, 상대에게 socket으로 전송할지 ###
udp = True 

class HandDetector:
    """
    Finds Hands using the mediapipe library. Exports the landmarks
    in pixel format. Adds extra functionalities like finding how
    many fingers are up or the distance between two fingers. Also
    provides bounding box info of the hand found.
    """

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, minTrackCon=0.5):
        """
        :param mode: In static mode, detection is done on each image: slower
        :param maxHands: Maximum number of hands to detect
        :param detectionCon: Minimum Detection Confidence Threshold
        :param minTrackCon: Minimum Tracking Confidence Threshold
        """
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []

    def findHands(self, img, draw=True, flipType=True):
        """
        Finds hands in a BGR image.
        :param img: Image to find the hands in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                ## lmList
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                ## bbox
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)

                ## draw
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                    # cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                    #               (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                    #               (255, 0, 255), 2)
                    cv2.putText(img, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 2)
        if draw:
            return allHands, img
        else:
            return allHands

    def fingersUp(self, myHand):
        """
        Finds how many fingers are open and returns in a list.
        Considers left and right hands separately
        :return: List of which fingers are up
        """
        myHandType = myHand["type"]
        myLmList = myHand["lmList"]
        if self.results.multi_hand_landmarks:
            fingers = []
            # Thumb
            if myHandType == "Right":
                if myLmList[self.tipIds[0]][0] > myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if myLmList[self.tipIds[0]][0] < myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # 4 Fingers
            for id in range(1, 5):
                if myLmList[self.tipIds[id]][1] < myLmList[self.tipIds[id] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img=None):
        """
        Find the distance between two landmarks based on their
        index numbers.
        :param p1: Point1
        :param p2: Point2
        :param img: Image to draw on.
        :param draw: Flag to draw the output on the image.
        :return: Distance between the points
                 Image with output drawn
                 Line information
        """

        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            return length, info, img
        else:
            return length, info


############################## SNAKE GAME LOGIC SECTION ##############################
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(cv2.CAP_PROP_FPS, 60)
fps = cap.get(cv2.CAP_PROP_FPS)

# color templates
red = (0, 0, 255) # red
megenta = (255, 0, 255) # magenta
green = (0, 255, 0) # green
yellow = (0, 255, 255) # yellow
cyan = (255, 255, 0) # cyan
detector = HandDetector(detectionCon=0.5, maxHands=1)


class SnakeGameClass:
    def __init__(self, pathFood, port_num, opp_ip, opp_port):
        self.points = []  # all points of the snake
        self.lengths = []  # distance between each point
        self.currentLength = 0  # total length of the snake
        self.allowedLength = 150  # total allowed Length
        self.previousHead = random.randint(100, 1000), random.randint(100, 600)

        self.speed = 0.1
        self.velocityX = random.choice([-1, 0, 1])
        self.velocityY = random.choice([-1, 0, 1])

        self.imgFood = cv2.imread(pathFood, cv2.IMREAD_UNCHANGED)
        self.hFood, self.wFood, _ = self.imgFood.shape
        self.foodPoint = 0, 0
        self.randomFoodLocation()

        self.score = 0
        self.gameOver = False

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', port_num))
        self.sock.settimeout(0.01)
        self.opp_addr = (opp_ip, opp_port)    

    # ---collision function---
    def ccw(self, p, a, b):
        vect_sub_ap = [a[0] - p[0], a[1] - p[1]]
        vect_sub_bp = [b[0] - p[0], b[1] - p[1]]
        return vect_sub_ap[0] * vect_sub_bp[1] - vect_sub_ap[1] * vect_sub_bp[0]

    def segmentIntersects(self, p1_a, p1_b, p2_a, p2_b):
        ab = self.ccw(p1_a, p1_b, p2_a) * self.ccw(p1_a, p1_b, p2_b)
        cd = self.ccw(p2_a, p2_b, p1_a) * self.ccw(p2_a, p2_b, p1_b)

        if (ab == 0 and cd == 0):
            if (p1_b[0] < p1_a[0] and p1_b[1] < p1_a[1]):
                p1_a, p1_b = p1_b, p1_a
            if (p2_b[0] < p2_a[0] and p2_b[1] < p2_a[1]):
                p2_a, p2_b = p2_b, p2_a
            return not ((p1_b[0] < p2_a[0] and p1_b[1] < p2_a[1]) or (p2_b[0] < p1_a[0] and p2_b[1] < p1_a[1]))

        return ab <= 0 and cd <= 0

    def isCollision(self, u1_head_pt, u2_pts):
        if not u2_pts:
            return False
        p1_a, p1_b = u1_head_pt[0], u1_head_pt[1]

        for u2_pt in u2_pts:
            p2_a, p2_b = u2_pt[0], u2_pt[1]
            if self.segmentIntersects(p1_a, p1_b, p2_a, p2_b):
                print(u2_pt)
                return True
        return False

    # ---collision function---end

    def randomFoodLocation(self):
        self.foodPoint = random.randint(100, 1000), random.randint(100, 600)

    def draw_snakes(self, imgMain, points, score, isMe):

        bodercolor = cyan
        maincolor = red

        if isMe:
            bodercolor = megenta
            maincolor = green
            # Draw Score
            cvzone.putTextRect(imgMain, f'Score: {score}', [0, 40],
                               scale=3, thickness=3, offset=10)

        # Draw Snake
        if points:
            cv2.circle(imgMain, points[-1][1], 20, bodercolor, cv2.FILLED)
            cv2.circle(imgMain, points[-1][1], 15, maincolor, cv2.FILLED)

        pts = np.array(points, np.int32)
        if len(pts.shape) == 3:
            pts = pts[:, 1]
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(imgMain, np.int32([pts]), False, maincolor, 15)

        return imgMain

    def draw_Food(self, imgMain):
        # Draw Food
        rx, ry = self.foodPoint
        imgMain = cvzone.overlayPNG(imgMain, self.imgFood,
                                    (rx - self.wFood // 2, ry - self.hFood // 2))
        return imgMain

    def my_snake_update(self, HandPoints, o_bodys):
        global opponent_data
        px, py = self.previousHead
        # ----HandsPoint moving ----
        s_speed = 30
        if HandPoints:
            m_x, m_y = HandPoints
            dx = m_x - px  # -1~1
            dy = m_y - py

            # speed 범위: 0~1460
            if math.hypot(dx, dy) > math.hypot(1280, 720) / 10:
                self.speed = math.hypot(1280, 720) / 10  # 146
            elif math.hypot(dx, dy) < s_speed:
                self.speed = s_speed
            else:
                self.speed = math.hypot(dx, dy)

            if dx != 0:
                self.velocityX = dx / 1280
            if dy != 0:
                self.velocityY = dy / 720

            # print(self.velocityX)
            # print(self.velocityY)

            cx = round(px + self.velocityX * self.speed)
            cy = round(py + self.velocityY * self.speed)

        else:
            self.speed = s_speed
            cx = round(px + self.velocityX * self.speed)
            cy = round(py + self.velocityY * self.speed)
        # ----HandsPoint moving ----end

        # print(f'{cx} , {cy}')

        self.points.append([[px, py], [cx, cy]])
        # print(f'{self.points}')

        distance = math.hypot(cx - px, cy - py)
        self.lengths.append(distance)
        # print(f'self.length -> {self.lengths}')
        self.currentLength += distance
        self.previousHead = cx, cy

        # Length Reduction
        if self.currentLength > self.allowedLength:
            for i, length in enumerate(self.lengths):
                self.currentLength -= length
                self.lengths = self.lengths[1:]
                self.points = self.points[1:]

                if self.currentLength < self.allowedLength:
                    break

        # Check if snake ate the Food
        rx, ry = self.foodPoint
        # print(f'먹이 위치 : {self.foodPoint}')
        if rx - self.wFood // 2 < cx < rx + self.wFood // 2 and \
                ry - self.hFood // 2 < cy < ry + self.hFood // 2:
            self.randomFoodLocation()
            self.allowedLength += 50
            self.score += 1
        if udp:
            send_data = str(cx) + '/' + str(cy) + '/' + str(self.points) + '/' + str(self.score)
            self.sock.sendto(send_data.encode(), self.opp_addr)

            try:
                data, _ = self.sock.recvfrom(10000)
                decode_data = data.decode()
                decode_data_list = decode_data.split('/')
                opponent_data['opp_head_x'] = int(decode_data_list[0])
                opponent_data['opp_head_y'] = int(decode_data_list[1])
                opponent_data['opp_body_node'] = eval(decode_data_list[2])
                opponent_data['opp_score'] = int(decode_data_list[3])

            except socket.timeout:
                pass
        else:
            socketio.emit('game_data', {'head_x': cx, 'head_y': cy, 'body_node': self.points, 'score': self.score, 'fps' : fps})

        # ---- Collision ----
        # print(self.points[-1])
        # print(self.points[:-5])
        if self.isCollision(self.points[-1], o_bodys):
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Hit")
            self.gameOver = False
            self.points = []  # all points of the snake
            self.lengths = []  # distance between each point
            self.currentLength = 0  # total length of the snake
            self.allowedLength = 150  # total allowed Length
            self.previousHead = 0, 0  # previous head point
            self.randomFoodLocation()

    def update(self, imgMain, receive_Data, HandPoints=[]):
        global gameover_flag

        if self.gameOver:
            # pass
            # cvzone.putTextRect(imgMain, "Game Over", [300, 400],
            #                    scale=7, thickness=5, offset=20)
            # cvzone.putTextRect(imgMain, f'Your Score: {self.score}', [300, 550],
            #                    scale=7, thickness=5, offset=20)
            gameover_flag = False
        else:
            # draw others snake
            o_body_node = []
            o_score = 0

            if receive_Data:
                o_body_node = receive_Data["opp_body_node"]
                o_score = receive_Data["opp_score"]

            # 0 이면 상대 뱀
            imgMain = self.draw_snakes(imgMain, o_body_node, o_score, 0)

            # update and draw own snake
            self.my_snake_update(HandPoints, o_body_node)
            imgMain = self.draw_Food(imgMain)
            # 1 이면 내 뱀
            imgMain = self.draw_snakes(imgMain, self.points, self.score, 1)

        return imgMain
    
    def __del__(self):
        self.sock.close()

opponent_data = {}
gameover_flag = False
######################################################################################

room_id = ""
sid = ""

MY_PORT = 0
game = SnakeGameClass("./static/food.png", MY_PORT, '0.0.0.0', 0)

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route("/enter_snake", methods=["GET", "POST"])
def enter_snake():
    global game
    global room_id
    global sid

    room_id = request.args.get('room_id')
    sid = request.args.get('sid')
    print(room_id, sid)

    return render_template("snake.html", room_id=room_id, sid=sid)


@socketio.on('connect')
def test_connect():
    print('Client connected!!!')


@socketio.on('disconnect')
def test_disconnect():
    global room_id
    global sid

    socketio.emit('server_disconnect', {'room_id' : room_id, 'sid' : sid})
    print('Client disconnected!!!')

@socketio.on('my_port')
def my_port(data):
    global MY_PORT

    MY_PORT = data['my_port']

# webpage로 부터 받은 상대방 주소 (socket 통신에 사용)
@socketio.on('opponent_address')
def set_address(data):
    global MY_PORT
    global game
    opp_ip = data['ip_addr']
    opp_port = data['port']

    game = SnakeGameClass("./static/food.png", MY_PORT, opp_ip, opp_port)

@socketio.on('opp_data_transfer')
def opp_data_transfer(data):
    global opponent_data
    opponent_data = data['data']
    # socketio.emit('opp_data_to_test_server', {'data' : data}, broadcast=True)
    # print('Received data from client:', opp_head_x, opp_head_y, opp_score, opp_sid)


@app.route('/snake')
def snake():
    def generate():
        global opponent_data
        global game
        global gameover_flag
        global sid

        time.sleep(1)

        while True:
            success, img = cap.read()
            img = cv2.flip(img, 1)
            hands, img = detector.findHands(img, flipType=False)

            # gaussian blur value
            # sigma = 10
            # img = (cv2.GaussianBlur(img, (0, 0), sigma))

            pointIndex = []

            if hands:
                lmList = hands[0]['lmList']
                pointIndex = lmList[8][0:2]

            img = game.update(img, opponent_data, pointIndex)

            # encode the image as a JPEG string
            _, img_encoded = cv2.imencode('.jpg', img)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n')

            if gameover_flag:
                print("game ended")
                gameover_flag = False
                time.sleep(1)
                socketio.emit('gameover', {'sid' : sid})
                time.sleep(2)
                break

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    socketio.run(app, host='localhost', port=5000, debug=True)
