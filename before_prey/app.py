########################################################################################################################
# KRAFTON JUNGLE 1기 나만의 무기 만들기 프로젝트
# Project Biam.io
# by.Team dabCAT
# 박찬우 : https://github.com/pcw999
# 박현우 : https://github.com/phwGithub
# 우한봄 : https://github.com/onebom
# 이민섭 : https://github.com/InFinity-dev
########################################################################################################################
##################################### PYTHON PACKAGE IMPORT ############################################################
import math
import random
import cvzone
import cv2
import numpy as np
import mediapipe as mp
import sys
import os
from flask_restful import Resource, Api
from flask_cors import CORS
from datetime import datetime
import datetime
import time
from flask import Flask, render_template, Response, request, redirect, url_for, session
from flask_socketio import SocketIO, emit, join_room
import socket
from engineio.payload import Payload
import simpleaudio as sa
import threading
import signal

# import pprint

########################################################################################################################
################################## SETTING GOLBAL VARIABLES ############################################################

Payload.max_decode_packets = 200

# PYTHON - ELECTRON VARIABLES
# This wil report the electron exe location, and not the /tmp dir where the exe
# is actually expanded and run from!
print(f"flask is running in {os.getcwd()}, __name__ is {__name__}", flush=True)
# print(f"flask/python env is {os.environ}", flush=True)
print(sys.version, flush=True)
# print(os.environ, flush=True)
# print(os.getcwd(), flush=True)
# print("User's Environment variable:")
# pprint.pprint(dict(os.environ), width = 1)

base_dir = '.'
if hasattr(sys, '_MEIPASS'):
    print('detected bundled mode', sys._MEIPASS)
    base_dir = os.path.join(sys._MEIPASS)

app = Flask(__name__, static_folder=os.path.join(base_dir, 'static'),
            template_folder=os.path.join(base_dir, 'templates'))

app.config['SECRET_KEY'] = "roomfitisdead"
app.config['DEBUG'] = True  # true will cause double load on startup
app.config['EXPLAIN_TEMPLATE_LOADING'] = False  # won't work unless debug is on

socketio = SocketIO(app, cors_allowed_origins='*')

CORS(app, origins='http://localhost:5000')

api = Api(app)

# Setting Path to food.png
pathFood = './static/food.png'

now_my_room = ""  # 현재 내가 있는 방
MY_PORT = 0  # socket_bind를 위한 내 포트 번호

############################################################ 아마도 자바스크립트로 HTML단에서 처리 예정
# 배경음악이나 버튼음은 자바스크립트, 게임오버나 스킬 사용 효과음은 파이썬
# Global Flag for BGM status
bgm_play_obj = None
# SETTING BGM PATH
bgm_path = './src-flask-server/static/bgm/main.wav'
vfx_1_path = './src-flask-server/static/bgm/curSelect.wav'
vfx_2_path = './src-flask-server/static/bgm/eatFood.wav'
vfx_3_path = './src-flask-server/static/bgm/boost.wav'
vfx_4_path = './src-flask-server/static/bgm/gameOver.wav'
vfx_5_path = './src-flask-server/static/bgm/stageWin.wav'


def play_bgm():
    global bgm_play_obj
    bgm_wave_obj = sa.WaveObject.from_wave_file(bgm_path)
    bgm_play_obj = bgm_wave_obj.play()
    bgm_play_obj.wait_done()


def stop_music_exit(signal, frame):
    global bgm_play_obj
    if bgm_play_obj is not None:
        bgm_play_obj.stop()
    exit(0)


def stop_bgm():
    global bgm_play_obj
    if bgm_play_obj is not None:
        bgm_play_obj.stop()


# Create a new thread for each sound effect selected by the user
def play_selected_sfx(track):
    sfx_wave_obj = sa.WaveObject.from_wave_file(track)
    sfx_play_obj = sfx_wave_obj.play()
    sfx_play_obj.wait_done()


# Create a thread for the BGM
bgm_thread = threading.Thread(target=play_bgm)
# Register the signal handler for SIGINT (Ctrl-C)
signal.signal(signal.SIGINT, stop_music_exit)


############################################################

########################################################################################################################
################################ Mediapipe Detecting Module ############################################################
class HandDetector:
    """
    Finds Hands using the mediapipe library. Exports the landmarks
    in pixel format. Adds extra functionalities like finding how
    many fingers are up or the distance between two fingers. Also
    provides bounding box info of the hand found.
    """

    def __init__(self, mode=False, maxHands=1, detectionCon=0.8, minTrackCon=0.5):
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

        return allHands

    # def drawHands(self, img):
    #     img2 = img.copy()
    #     if self.results.multi_hand_landmarks:
    #         for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
    #             self.mpDraw.draw_landmarks(img2, handLms, self.mpHands.HAND_CONNECTIONS)
    #     return img2

    # Only Draw Index Finger
    def drawHands(self, img):
        img2 = img.copy()
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                # Access landmarks of the index finger
                index_finger_landmarks = handLms.landmark[self.mpHands.HandLandmark.INDEX_FINGER_TIP]
                # Draw a circle at the index finger tip landmark
                cv2.circle(img2, (
                    int(index_finger_landmarks.x * img2.shape[1]), int(index_finger_landmarks.y * img2.shape[0])), 27,
                           (255, 0, 255), 3)
        return img2

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


########################################################################################################################
################################## SNAKE GAME LOGIC SECTION ############################################################
# video setting
cap = cv2.VideoCapture(0)

# Ubuntu YUYV cam setting low frame rate problem fixed
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

cap.set(3, 1280)
cap.set(4, 720)
cap.set(cv2.CAP_PROP_FPS, 30)  # TODO : 영향 확인하기, 시간 탐지 기법 중 하나가 프레임이라 프레임 맞춰줌
fps = cap.get(cv2.CAP_PROP_FPS)

# Color templates
red = (0, 0, 255)  # red
megenta = (255, 0, 255)  # magenta
green = (0, 255, 0)  # green
yellow = (0, 255, 255)  # yellow
cyan = (255, 255, 0)  # cyan
detector = HandDetector(detectionCon=0.5, maxHands=1)

class MultiGameClass:
    # 생성자, class를 선언하면서 기본 변수들을 설정함
    def __init__(self, pathFood):
        self.points = []  # all points of the snake
        self.lengths = []  # distance between each point
        self.currentLength = 0  # total length of the snake
        self.allowedLength = 150  # total allowed Length
        self.previousHead = 0, 0
        self.score = 0

        self.speed = 5
        self.minspeed = 10
        self.maxspeed = math.hypot(1280, 720) / 10
        self.velocityX = random.choice([-1, 0, 1])
        self.velocityY = random.choice([-1, 1])

        self.imgFood = cv2.imread(pathFood, cv2.IMREAD_UNCHANGED)
        self.hFood, self.wFood, _ = self.imgFood.shape
        self.foodPoint = 640, 360
  
        self.opp_score = 0
        self.opp_points = []
        self.dist = 500

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.opp_addr = ()
        self.udp_count = 0
        self.user_number = 0

        self.is_udp = False
        self.foodOnOff = True
        self.user_move = False
        self.check_collision = False
        self.gen = True

    # 통신 관련 변수 설정
    def set_socket(self, my_port, opp_ip, opp_port):
        self.sock.bind(('0.0.0.0', int(my_port)))
        self.sock.settimeout(0.01)  # TODO 만약 udp, 서버 선택 오류 시 다시 0.02로
        self.opp_addr = (opp_ip, int(opp_port))

    # udp로 통신할지 말지
    def test_connect(self, sid):
        a = 0
        b = 0
        test_code = str(sid)

        for i in range(50):
            if i % 2 == 0:
                test_code = str(sid)
            self.sock.sendto(test_code.encode(), self.opp_addr)
            try:
                data, _ = self.sock.recvfrom(600)
                test_code = data.decode()
                if test_code == str(sid):
                    b += 1
            except socket.timeout:
                a += 1

        if a != 50 and b != 0:
            self.is_udp = True

        print(f"connection MODE : {self.is_udp} / a = {a}, b = {b}")
        socketio.emit('NetworkMode', {'UDP': self.is_udp})
        socketio.emit('game_ready')

    # 송출될 프레임 업데이트
    def update(self, imgMain, HandPoints):
        self.my_snake_update(HandPoints)

        if self.is_udp:
            self.receive_data_from_opp()

        imgMain = self.draw_Food(imgMain)

        # 1 이면 내 뱀 / 0 이면 상대 뱀
        imgMain = self.draw_snakes(imgMain, self.points, self.score, 1)
        imgMain = self.draw_snakes(imgMain, self.opp_points, self.opp_score, 0)

        # ---head와 handsPoint 점선으로 잇기---
        # for p in np.linspace(self.previousHead, HandPoints, 10):
        #     cv2.circle(imgMain, tuple(np.int32(p)), 2, (255, 0, 255), -1)

        self.send_data_to_opp()

        if self.check_collision:
            if self.isCollision(self.points[-1], self.opp_points):
                self.execute()

        return imgMain
    
    # 내 뱀 상황 업데이트
    def my_snake_update(self, HandPoints):
        px, py = self.previousHead

        s_speed = 30
        cx, cy = self.set_snake_speed(HandPoints, s_speed)
        socketio.emit('finger_cordinate', {'head_x': cx, 'head_y': cy})

        self.points.append([[px, py], [cx, cy]])

        distance = math.hypot(cx - px, cy - py)
        self.lengths.append(distance)
        self.currentLength += distance
        self.previousHead = cx, cy

        self.length_reduction()

        if self.foodOnOff:
            self.check_snake_eating(cx, cy)

        if self.opp_points:
            self.dist = ((self.points[-1][1][0] - self.opp_points[-1][1][0]) ** 2 + (
                    self.points[-1][1][1] - self.opp_points[-1][1][1]) ** 2) ** 0.5
        # 할일: self.multi가 false일 때, pt_dist html에 보내기
        # print(f"point distance: {pt_dist}")
        socketio.emit('h2h_distance', self.dist)

    # 뱀 속도 설정
    def set_snake_speed(self, HandPoints, s_speed):
        px, py = self.previousHead
        # ----HandsPoint moving ----
        s_speed = 20
        if HandPoints:
            m_x, m_y = HandPoints
            dx = m_x - px  # -1~1
            dy = m_y - py

            # speed 범위: 0~1460
            if math.hypot(dx, dy) > self.maxspeed:  # 146
                self.speed = self.maxspeed
            elif math.hypot(dx, dy) < self.minspeed:
                self.speed = self.minspeed
            else:
                self.speed = math.hypot(dx, dy)

            if dx != 0:
                self.velocityX = dx / 1280
            if dy != 0:
                self.velocityY = dy / 720

            # print(self.velocityX)
            # print(self.velocityY)

        else:
            self.speed = self.minspeed

        cx = round(px + self.velocityX * self.speed)
        cy = round(py + self.velocityY * self.speed)
        # ----HandsPoint moving ----end
        if cx < 0 or cx > 1280 or cy < 0 or cy > 720:
            if cx < 0: cx = 0
            if cx > 1280: cx = 1280
            if cy < 0: cy = 0
            if cy > 720: cy = 720

        if cx == 0 or cx == 1280:
            self.velocityX = -self.velocityX
        if cy == 0 or cy == 720:
            self.velocityY = -self.velocityY

        return cx, cy

    # 뱀 길이 조정    
    def length_reduction(self):
        if self.currentLength > self.allowedLength:
            for i, length in enumerate(self.lengths):
                self.currentLength -= length
                self.lengths = self.lengths[1:]
                self.points = self.points[1:]

                if self.currentLength < self.allowedLength:
                    break

    # 뱀 식사 여부 확인
    def check_snake_eating(self, cx, cy):
        rx, ry = self.foodPoint
        if (rx - (self.wFood // 2) < cx < rx + (self.wFood // 2)) and (
                ry - (self.hFood // 2) < cy < ry + (self.hFood // 2)):
            # sfx_thread = threading.Thread(target=play_selected_sfx, args=(vfx_2_path,))
            # sfx_thread.start()
            self.allowedLength += 50
            self.score += 1

            self.foodOnOff = False
            socketio.emit('user_ate_food', {'score': self.score})

    # 먹이 그려주기
    def draw_Food(self, imgMain):
        rx, ry = self.foodPoint
        socketio.emit('foodPoint', {'food_x': rx, 'food_y': ry})
        imgMain = cvzone.overlayPNG(imgMain, self.imgFood, (rx - self.wFood // 2, ry - self.hFood // 2))

        return imgMain
    
    # 데이터 수신 (udp 통신 일때만 사용)
    def receive_data_from_opp(self):
        try:
            data, _ = self.sock.recvfrom(15000)
            decode_data = data.decode()
            if decode_data[0] == '[':
                self.opp_points = eval(decode_data)
                self.udp_count = 0
            else:
                test_code = decode_data
                self.sock.sendto(test_code.encode(), self.opp_addr)
        except socket.timeout:
            self.udp_count += 1
            if self.udp_count > 25:
                socketio.emit('opponent_escaped')

    # 뱀 그려주기
    def draw_snakes(self, imgMain, points, score, isMe):

        bodercolor = cyan
        maincolor = red

        if isMe:
            bodercolor = megenta
            maincolor = green
            # Draw Score
            # cvzone.putTextRect(imgMain, f'Score: {score}', [0, 40],
            #                    scale=3, thickness=3, offset=10)

        # Change hue every 100ms
        change_interval = 100

        hue = int(time.time() * change_interval % 180)  # TODO : 마지막에 성능 부족 시 아낄 수 있음
        rainbow = np.array([hue, 255, 255], dtype=np.uint8)
        rainbow = cv2.cvtColor(np.array([[rainbow]]), cv2.COLOR_HSV2BGR)[0, 0]
        # Convert headcolor to tuple of integers
        rainbow = tuple(map(int, rainbow))

        # Draw Snake
        # TODO : 아이템 먹으면 무지개 색으로 변하게?
        pts = np.array(points, np.int32)
        if len(pts.shape) == 3:
            pts = pts[:, 1]
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(imgMain, np.int32([pts]), False, maincolor, 15)

        if points:
            cv2.circle(imgMain, points[-1][1], 20, bodercolor, cv2.FILLED)
            cv2.circle(imgMain, points[-1][1], 15, rainbow, cv2.FILLED)

        return imgMain

    # 데이터 전송
    def send_data_to_opp(self):
        if self.is_udp:
            data_set = str(self.points)
            self.sock.sendto(data_set.encode(), self.opp_addr)
        else:
            socketio.emit('game_data', {'body_node': self.points})

    # 세 개의 점 방향성
    def ccw(self, p, a, b):
        s = p[0] * a[1] + a[0] * b[1] + b[0] * p[1]
        s -= (p[1] * a[0] + a[1] * b[0] + b[1] * p[0])

        if s > 0:
            return 1
        elif s == 0:
            return 0
        else:
            return -1

    # 두 선분의 교차 판단
    def segmentIntersects(self, p1_a, p1_b, p2_a, p2_b):
        ab = self.ccw(p1_a, p1_b, p2_a) * self.ccw(p1_a, p1_b, p2_b)
        cd = self.ccw(p2_a, p2_b, p1_a) * self.ccw(p2_a, p2_b, p1_b)

        if (ab == 0 and cd == 0):
            if (p1_a[0] > p1_b[0] or p1_a[1] > p1_b[1]):
                p1_a, p1_b = p1_b, p1_a
            if (p2_a[0] > p2_b[0] or p2_a[1] > p2_b[1]):
                p2_a, p2_b = p2_b, p2_a
            return (p2_a[0] <= p1_b[0] and p2_a[1] <= p1_b[1]) and (p1_a[0] <= p2_b[0] and p1_a[1] <= p2_b[1])

        return ab <= 0 and cd <= 0

    # 충돌 판단
    def isCollision(self, u1_head_pt, u2_pts):
        if not u2_pts:
            return False
        # p1_a, p1_b = np.array(u1_head_pt[0]), np.array(u1_head_pt[1]) # p1_b: head point
        p1_a, p1_b = u1_head_pt[0], u1_head_pt[1]

        for u2_pt in u2_pts:
            # p2_a, p2_b = np.array(u2_pt[0]), np.array(u2_pt[1])
            p2_a, p2_b = u2_pt[0], u2_pt[1]

            if self.segmentIntersects(p1_a, p1_b, p2_a, p2_b):
                # print(p1_a, p1_b, p2_a, p2_b)
                return True

        return False

    # 뱀이 충돌했을때
    def execute(self):
        self.check_collision = False
        self.user_move = False
        socketio.emit('gameover')

    # 소멸자 소켓 bind 해제
    def __del__(self):
        self.sock.close()

########################################################################################################################
######################################## FLASK APP ROUTINGS ############################################################

multi = MultiGameClass(pathFood)


# Defualt Root Routing for Flask Server Check
@api.resource('/')
class HelloWorld(Resource):
    def get(self):
        print(f'Electron GET Requested from HTML', flush=True)
        data = {'Flask 서버 클라이언트 to Electron': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        return data


# Game Main Menu
@app.route("/index", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route('/testbed')
def testbed():
    return render_template("testbed.html")


@app.route('/mazerunner')
def mazerunner():
    return render_template("mazerunner.html")


# Game Screen
@app.route("/enter_snake", methods=["GET", "POST"])
def enter_snake():
    global now_my_room
    global multi

    multi = MultiGameClass(pathFood)

    now_my_room = request.args.get('room_id')
    multi.user_number = int(request.args.get('user_num'))

    return render_template("snake.html", room_id=now_my_room)


########################################################################################################################
############## SERVER SOCKET AND PEER TO PEER ESTABLISHMENT ############################################################

# 페이지에서 로컬 flask 서버와 소켓 통신 개시 되었을때 자동으로 실행
@socketio.on('connect')
def test_connect():
    print('Client connected!!!')


# 페이지에서 로컬 flask 서버와 소켓 통신 종료 되었을때 자동으로 실행
@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected!!!')


# 현재 내 포트 번호 요청
@socketio.on('my_port')
def my_port(data):
    global MY_PORT

    MY_PORT = data['my_port']


# webpage로 부터 받은 상대방 주소 (socket 통신에 사용)
@socketio.on('opponent_address')
def set_address(data):
    global MY_PORT
    global multi

    opp_ip = data['ip_addr']
    opp_port = data['port']
    sid = request.sid

    multi.set_socket(MY_PORT, opp_ip, opp_port)
    multi.test_connect(sid)


# socketio로 받은 상대방 정보
@socketio.on('opp_data_transfer')
def opp_data_transfer(data):
    global multi
    multi.opp_points = data['opp_body_node']


# socketio로 받은 먹이 위치
@socketio.on('set_food_location')
def set_food_loc(data):
    global multi
    multi.foodPoint = data['foodPoint']
    multi.foodOnOff = True


# socketio로 받은 먹이 위치와 상대 점수
@socketio.on('set_food_location_score')
def set_food_loc(data):
    global multi
    multi.foodPoint = data['foodPoint']
    multi.opp_score = data['opp_score']
    multi.foodOnOff = True


########################################################################################################################
######################################## MAIN GAME ROUNTING ############################################################
@app.route('/snake')
def snake():
    def generate():
        global multi

        if multi.user_number == 1:
            start_cx = 100
            start_cy = 360
            multi.previousHead = (100, 360)
        elif multi.user_number == 2:
            start_cx = 1180
            start_cy = 360
            multi.previousHead = (1180, 360)
        else:
            start_cx, start_cy = 640, 360

        while True:
            success, img = cap.read()
            img = cv2.flip(img, 1)

            hands = detector.findHands(img, flipType=False)
            img = detector.drawHands(img)

            pointIndex = []

            if hands and multi.user_move:
                lmList = hands[0]['lmList']
                pointIndex = lmList[8][0:2]
            if not multi.user_move:
                pointIndex = [start_cx, start_cy]

            if not multi.user_move:
                if multi.user_number == 1:
                    start_cx += 5
                    if start_cx > 350:
                        start_cx = 70
                        multi.user_move = True
                        multi.check_collision = True
                elif multi.user_number == 2:
                    start_cx -= 5
                    if start_cx < 930:
                        start_cx = 1210
                        multi.user_move = True
                        multi.check_collision = True

            img = multi.update(img, pointIndex)

            # encode the image as a JPEG string
            _, img_encoded = cv2.imencode('.jpg', img)
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n')

            if not multi.gen:
                break

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

########################################################################################################################
########################## Legacy Electron Template Routing ############################################################
@app.route('/hello')
def hello():
    return render_template('hello.html', msg="YOU")


@app.route('/hello-vue')
def hello_vue():
    return render_template('hello-vue.html', msg="WELCOME 🌻")


########################################################################################################################
####################################### FLASK APP ARGUMENTS ############################################################

if __name__ == "__main__":
    socketio.run(app, host='localhost', port=5000, debug=False, allow_unsafe_werkzeug=True)

########################################################################################################################
