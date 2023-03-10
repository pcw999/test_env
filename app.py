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
from io import StringIO

from src.maze_manager import MazeManager

import simpleaudio as sa
import threading
import signal

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
current_dir = os.path.abspath(os.path.dirname(__file__))

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
pathFood = os.path.join(current_dir, 'static', 'food.png')

opponent_data = {}  # 상대 데이터 (현재 손위치, 현재 뱀위치)
gameover_flag = False  # ^^ 게임오버
bot_flag = False
now_my_room = ""  # 현재 내가 있는 방
now_my_sid = ""  # 현재 나의 sid
MY_PORT = 0  # socket_bind를 위한 내 포트 번호
user_number = 0  # 1p, 2p를 나타내는 번호
user_move = False
game_over_for_debug = False
start = False

############################################################ 아마도 자바스크립트로 HTML단에서 처리 예정
# 배경음악이나 버튼음은 자바스크립트, 게임오버나 스킬 사용 효과음은 파이썬
# Global Flag for BGM status
bgm_play_obj = None
# SETTING BGM PATH
bgm_path = os.path.join(current_dir, 'static', 'bgm','main.wav')
sfx_1_path = os.path.join(current_dir, 'static', 'bgm','curSelect.wav') 
sfx_2_path = os.path.join(current_dir, 'static', 'bgm','eatFood.wav') 
sfx_3_path = os.path.join(current_dir, 'static', 'bgm','skill.wav')
sfx_4_path = os.path.join(current_dir, 'static', 'bgm','gameOver.wav')  
sfx_5_path = os.path.join(current_dir, 'static', 'bgm','gameWin.wav') 
sfx_6_path = os.path.join(current_dir, 'static', 'bgm','warning.wav') 
sfx_7_path = os.path.join(current_dir, 'static', 'bgm','dead.wav') 


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

    def __init__(self, mode=False, maxHands=1, detectionCon=0.5, minTrackCon=0.8):
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

class SnakeGameClass:
    # 생성자, class를 선언하면서 기본 변수들을 설정함
    def __init__(self, pathFood):
        self.points = []  # all points of the snake
        self.lengths = []  # distance between each point
        self.currentLength = 0  # total length of the snake
        self.allowedLength = 150  # total allowed Length
        self.previousHead = (int), (int)  # TODO 이거 됨 ?

        self.speed = 5
        self.minspeed = 10
        self.maxspeed = math.hypot(1280, 720) / 10
        self.velocityX = random.choice([-1, 0, 1])
        self.velocityY = random.choice([-1, 1])

        self.imgFood = cv2.imread(pathFood, cv2.IMREAD_UNCHANGED)
        self.hFood, self.wFood, _ = self.imgFood.shape
        self.foodPoint = 640, 360

        self.score = 0
        self.bestScore = 0

        self.opp_score = 0
        self.opp_addr = ()
        self.is_udp = False
        self.udp_count = 0
        self.foodOnOff = True
        self.multi = False

        self.maze_start = [[], []]
        self.maze_end = [[], []]
        self.maze_map = np.array([])
        self.passStart = False
        self.passMid = False
        self.maze_img = np.array([0])
        self.dist = 500

        self.menu_type = 0
        self.menu_time = 0
        self.line_flag = False

    def global_intialize(self):
        global user_number
        self.points = []  # all points of the snake
        self.lengths = []  # distance between each point
        self.currentLength = 0  # total length of the snake
        self.allowedLength = 150  # total allowed Length
        self.previousHead = (int), (int)  # TODO 이거 됨 ?

        self.speed = 5
        self.minspeed = 10
        self.maxspeed = math.hypot(1280, 720) / 10
        self.velocityX = random.choice([-1, 0, 1])
        self.velocityY = random.choice([-1, 1])

        self.hFood, self.wFood, _ = self.imgFood.shape
        self.foodPoint = 640, 360
        self.foodtimeLimit = 0

        self.score = 0
        self.opp_score = 0
        self.opp_addr = ()
        self.is_udp = False
        self.udp_count = 0
        self.foodOnOff = True
        self.multi = False

        self.timer_end = 0
        self.maze_start = [[], []]
        self.maze_end = [[], []]
        self.maze_map = np.array([])
        self.passStart = False
        user_number = 0

    def ccw(self, p, a, b):
        s = p[0] * a[1] + a[0] * b[1] + b[0] * p[1]
        s -= (p[1] * a[0] + a[1] * b[0] + b[1] * p[0])

        if s > 0:
            return 1
        elif s == 0:
            return 0
        else:
            return -1

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

    def maze_collision(self, head_pt, previous_pt):
        head_pt = np.array(head_pt).astype(int)
        # if self.maze_map[int(head_pt[1]),int(head_pt[0])]==1:
        #   return True
        pt_a = np.array(previous_pt).astype(int)
        line_norm = np.linalg.norm(pt_a - head_pt).astype(int)
        points_on_line = np.linspace(pt_a, head_pt, line_norm)
        for p in points_on_line:
            try:
                if self.maze_map[int(p[1]), int(p[0])] == 1:
                    return True
            except:
                pass

        return False

    # maze 초기화
    def maze_initialize(self):
        global bot_flag
        bot_flag = False
        self.maze_start, self.maze_mid, self.maze_end, self.maze_map = create_maze(720 - 300, 1280 - 300, 5, 12)
        self.maze_map = np.pad(self.maze_map, ((150, 150), (150, 150)), 'constant', constant_values=0)
        self.maze_img = self.create_maze_image()

        self.previousHead = (0, 360)
        self.velocityX = 0
        self.velocityY = 0
        self.points = []
        self.maxspeed = math.hypot(1280, 720) / 10
        self.passStart = False
        self.passMid = False
        self.line_flag = True
        self.timer_end = time.time() + 91

    def menu_initialize(self):
        global bot_flag
        bot_flag = False
        self.previousHead = (0, 360)
        self.velocityX = 0
        self.velocityY = 0
        self.line_flag = True
        self.points = []

    def testbed_initialize(self):
        global bot_data
        self.previousHead = (0, 360)
        self.velocityX = 0
        self.velocityY = 0
        self.points = []
        self.foodOnOff = True
        self.multi = False

        bot_data = {'bot_head_x': 1000,
                    'bot_head_y': 360,
                    'bot_body_node': [],
                    'currentLength': 0,
                    'lengths': [],
                    'bot_velocityX': random.choice([-1, 1]),
                    'bot_velocityY': random.choice([-1, 1])}

    def draw_snakes(self, imgMain, points, HandPoints, isMe):
        global bot_flag

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

        # Single Mod Collision Padding Design
        if bot_flag and isMe:
            cv2.polylines(imgMain, np.int32([pts]), False, red, 25)
            cv2.polylines(imgMain, np.int32([pts]), False, maincolor, 15)
        else:
            cv2.polylines(imgMain, np.int32([pts]), False, maincolor, 15)

        if isMe and HandPoints and self.line_flag:
            for p in np.linspace(self.previousHead, HandPoints, 10):
                cv2.circle(imgMain, tuple(np.int32(p)), 5, (0, 255, 0), -1)

        if points:
            cv2.circle(imgMain, points[-1][1], 20, bodercolor, cv2.FILLED)
            cv2.circle(imgMain, points[-1][1], 15, rainbow, cv2.FILLED)

        return imgMain

    def draw_Food(self, imgMain):
        rx, ry = self.foodPoint
        socketio.emit('foodPoint', {'food_x': rx, 'food_y': ry})
        imgMain = cvzone.overlayPNG(imgMain, self.imgFood, (rx - self.wFood // 2, ry - self.hFood // 2))

        return imgMain

    ############################################################
    def create_maze_image(self):
        img = np.zeros((720, 1280, 3), dtype=np.uint8)

        # White Wall
        img[np.where(self.maze_map == 1)] = (255, 255, 255)
        # Start Green
        img[np.where(self.maze_map == 2)] = (0, 255, 0)
        # End Red
        img[np.where(self.maze_map == 3)] = (0, 0, 255)
        # mid point
        # img[np.where(self.maze_map == 4)] = (255, 0, 0)
        return img

    # 내 뱀 상황 업데이트 - main에서
    def my_snake_update_menu(self, HandPoints):
        px, py = self.previousHead
        s_speed = 30
        cx, cy = self.set_snake_speed(HandPoints, s_speed)

        self.points.append([[px, py], [cx, cy]])
        distance = math.hypot(cx - px, cy - py)
        self.lengths.append(distance)
        self.currentLength += distance
        self.previousHead = cx, cy

        self.length_reduction()

        menu_type = 0
        # hover event emit 할 필요 TODO
        if 490 <= cx <= 790:
            if 70 <= cy <= 170:  # menu_type: 2, SINGLE PLAY
                menu_type = 2
            elif 310 <= cy <= 410:  # menu_type: 1, MULTI PLAY
                menu_type = 1
            elif 550 <= cy <= 650:  # menu_type: 3, MAZE RUNNER
                menu_type = 3

        return menu_type

    # 내 뱀 상황 업데이트 - maze play에서
    def my_snake_update_mazeVer(self, HandPoints):
        px, py = self.previousHead
        s_speed = 30
        cx, cy = self.set_snake_speed(HandPoints, s_speed)

        self.points.append([[px, py], [cx, cy]])
        distance = math.hypot(cx - px, cy - py)
        self.lengths.append(distance)
        self.currentLength += distance
        self.previousHead = cx, cy

        self.length_reduction()
        if self.maze_collision([cx, cy], [px, py]):
            self.passStart = False
            self.passMid = False
            self.previousHead = 0, 360
            self.points = []
            self.lengths = []
            self.currentLength = 0

        # start point 시작!
        start_pt1, start_pt2 = self.maze_start
        if (start_pt1[0] <= cx <= start_pt2[0]) and (start_pt1[1] <= cy <= start_pt2[1]):
            self.passStart = True

        # 중간 point 패스!
        mid_pt1, mid_pt2 = self.maze_mid
        if (mid_pt1[0] <= cx <= mid_pt2[0]) and (mid_pt1[1] <= cy <= mid_pt2[1]):
            if self.passStart:
                self.passMid = True

        # end point 도달
        end_pt1, end_pt2 = self.maze_end
        # print(f"end point : 1-{end_pt1}, 2-{end_pt2}")
        if (end_pt1[0] <= cx <= end_pt2[0]) and (end_pt1[1] <= cy <= end_pt2[1]):
            if self.passStart and self.passMid:
                self.maze_initialize()

    # 내 뱀 상황 업데이트
    def my_snake_update(self, HandPoints, opp_bodys):
        global bot_flag
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

        self.send_data_to_html()

        if opp_bodys:
            self.dist = ((self.points[-1][1][0] - opp_bodys[-1][1][0]) ** 2 + (
                    self.points[-1][1][1] - opp_bodys[-1][1][1]) ** 2) ** 0.5
        # 할일: self.multi가 false일 때, pt_dist html에 보내기
        # print(f"point distance: {pt_dist}")
        socketio.emit('h2h_distance', self.dist)

        opp_bodys_collsion = opp_bodys

        # Single Play Self Collision
        if bot_flag:
            opp_bodys_collsion = opp_bodys + self.points[:-3]

        if self.isCollision(self.points[-1], opp_bodys_collsion):
            global user_move
            global gameover_flag
            sfx_thread = threading.Thread(target=play_selected_sfx, args=(sfx_7_path,))
            sfx_thread.start()
            gameover_flag = True
            if user_move:
                self.execute()

    ################################## VECTORING SPEED METHOD ##########################################################
    # def set_snake_speed(self, HandPoints, s_speed):
    #   px, py = self.previousHead
    #   # ----HandsPoint moving ----
    #   if HandPoints:
    #       m_x, m_y = HandPoints
    #       dx = m_x - px  # -1~1
    #       dy = m_y - py
    #
    #
    #       # head로부터 handpoint가 근접하면 이전 direction을 따름
    #       if math.hypot(dx, dy) < 5:
    #           self.speed=5 # 최소속도
    #       else:
    #           if math.hypot(dx, dy) > 50:
    #               self.speed=50 #최대속도
    #           else:
    #               self.speed = math.hypot(dx, dy)
    #
    #       # 벡터 합 생성,크기가 1인 방향 벡터
    #       if dx!=0 and dy!=0:
    #         self.velocityX = dx/math.sqrt(dx**2+dy**2)
    #         self.velocityY = dy/math.sqrt(dx**2+dy**2)
    #
    #   else:
    #       self.speed=5
    #
    #   cx = round(px + self.velocityX*self.speed)
    #   cy = round(py + self.velocityY*self.speed)
    #   # ----HandsPoint moving ----end
    #   if cx < 0 or cx > 1280 or cy < 0 or cy > 720:
    #     if cx < 0: cx = 0
    #     if cx > 1280: cx = 1280
    #     if cy < 0: cy = 0
    #     if cy > 720: cy = 720
    #
    #   if cx == 0 or cx == 1280:
    #     self.velocityX = -self.velocityX
    #   if cy == 0 or cy == 720:
    #     self.velocityY = -self.velocityY
    #
    #   return cx, cy
    ####################################################################################################################

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
            sfx_thread = threading.Thread(target=play_selected_sfx, args=(sfx_2_path,))
            sfx_thread.start()
            self.allowedLength += 50
            self.score += 1

            if self.multi:
                self.foodOnOff = False
                socketio.emit('user_ate_food', {'score': self.score})
            else:
                if self.score > self.bestScore:
                    self.bestScore = self.score
                    socketio.emit('bestScore', {'bestScore': self.bestScore})

                single_game.foodtimeLimit = time.time() + 11
                self.foodPoint = random.randint(100, 1000), random.randint(100, 600)

    # 뱀이 충돌했을때
    def execute(self):
        global user_number
        global user_move
        global game_over_for_debug
        # self.points = []  # all points of the snake
        # self.lengths = []  # distance between each point
        # self.currentLength = 0  # total length of the snake
        # self.allowedLength = 150  # total allowed Length
        # self.score = 0
        # self.previousHead = 0, 360
        user_move = False
        game_over_for_debug = True
        socketio.emit('gameover')

    def update_mazeVer(self, imgMain, HandPoints):
        self.my_snake_update_mazeVer(HandPoints)
        imgMain = self.draw_snakes(imgMain, self.points, HandPoints, 1)

        return imgMain

    # 송출될 프레임 업데이트
    def update(self, imgMain, HandPoints):
        global opponent_data

        opp_bodys = []
        # 0 이면 상대 뱀
        if opponent_data:
            opp_bodys = opponent_data['opp_body_node']
        imgMain = self.draw_snakes(imgMain, opp_bodys, HandPoints, 0)

        # update and draw own snake
        self.my_snake_update(HandPoints, opp_bodys)
        imgMain = self.draw_Food(imgMain)
        # 1 이면 내 뱀
        imgMain = self.draw_snakes(imgMain, self.points, HandPoints, 1)

        return imgMain

    # Menu 화면에서 쓰일 검은 배경 뱀
    def update_blackbg(self, imgMain, HandPoints):
        global gameover_flag, opponent_data

        # update and draw own snake
        menu_type = self.my_snake_update_menu(HandPoints)

        if self.menu_type != 0:
            if self.menu_type == menu_type:
                self.menu_time += 1

            if self.menu_time == 30:  # 5초간 menu bar에 머무른 경우
                # 할일: menu_type(1:multi, 2:single, 3:maze) 사용해서 routing
                socketio.emit("selected_menu_type", {'menu_type': self.menu_type})
                self.menu_time = 0
                self.menu_type = 0

        self.menu_type = menu_type

        imgMain = self.draw_snakes(imgMain, self.points, HandPoints, 1)

        return imgMain

    def send_data_to_html(self):
        socketio.emit('game_data_for_debug', {'score': self.score, 'fps': fps})

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
        self.cut_idx = 0

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.opp_addr = ()
        self.udp_count = 0
        self.user_number = 0
        self.queue = []

        self.is_udp = False
        self.foodOnOff = True
        self.user_move = False
        self.check_collision = False
        self.gen = True
        self.skill_flag = False
        self.opp_skill_flag = False
        self.line_flag = False

    # 통신 관련 변수 설정
    def set_socket(self, my_port, opp_ip, opp_port):
        self.sock.bind(('0.0.0.0', int(my_port)))
        self.sock.settimeout(0.02)  # TODO 만약 udp, 서버 선택 오류 시 다시 0.02로
        self.opp_addr = (opp_ip, int(opp_port))

    # udp로 통신할지 말지
    def test_connect(self, sid):
        missing_cnt = 0
        self_sid_cnt = 0
        test_code = str(sid)

        if test_code == "1":
            for i in range(50):
                if i % 3 == 0 and self_sid_cnt == 0:
                    test_code = '1'
                self.sock.sendto(test_code.encode(), self.opp_addr)
                try:
                    data, _ = self.sock.recvfrom(100)
                    test_code = data.decode()
                    if test_code == str(sid):
                        test_code = '2'
                        self_sid_cnt += 1
                        if self_sid_cnt > 3:
                            break
                except socket.timeout:
                    missing_cnt += 1

        elif test_code == "2":
            for i in range(50):
                if i % 3 == 0 and self_sid_cnt == 0:
                    test_code = '2'
                self.sock.sendto(test_code.encode(), self.opp_addr)
                try:
                    data, _ = self.sock.recvfrom(100)
                    test_code = data.decode()
                    if test_code == str(sid):
                        test_code = '1'
                        self_sid_cnt += 1
                        if self_sid_cnt > 3:
                            break
                except socket.timeout:
                    missing_cnt += 1

        self.con_cnt = (missing_cnt // 3) + 1

        # 상대로 부터 받은 본인 Player Number 카운터가 1보다 클때 UDP 연결
        if self_sid_cnt > 1 and missing_cnt < 26:
            # 시연 위해 UDP 연결 비활성화
            self.is_udp = True
            self.sock.settimeout(0.01)
            # Flushing socket buffer
            for _ in range(50):
                self.sock.recv(0)
            self.sock.settimeout(0)

        print(f"connection MODE : {self.is_udp} / missing_cnt = {missing_cnt}, self_sid_cnt = {self_sid_cnt}")
        socketio.emit('NetworkMode', {'UDP': self.is_udp})
        socketio.emit('game_ready')

    # 송출될 프레임 업데이트
    def update(self, imgMain, HandPoints):
        self.my_snake_update(HandPoints)

        if self.is_udp:
            self.receive_data_from_opp()

        imgMain = self.draw_Food(imgMain)

        # 1 이면 내 뱀 / 0 이면 상대 뱀
        imgMain = self.draw_snakes(imgMain, self.points, HandPoints, 1)
        imgMain = self.draw_snakes(imgMain, self.opp_points, HandPoints, 0)

        self.send_data_to_opp()

        if self.check_collision and self.points:
            coll_bool = self.isCollision(self.points[-1], self.opp_points)
            if coll_bool:
                if self.skill_flag:
                    socketio.emit("opp_cut_idx", {"cut_idx": coll_bool})
                    self.skill_flag = False
                else:
                    sfx_thread = threading.Thread(target=play_selected_sfx, args=(sfx_7_path,))
                    sfx_thread.start()
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

        if self.opp_points and self.points:
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
            sfx_thread = threading.Thread(target=play_selected_sfx, args=(sfx_2_path,))
            sfx_thread.start()
            self.allowedLength += 50
            self.score += 1

            self.foodOnOff = False
            socketio.emit('user_ate_food', {'score': self.score})

            if self.score % 5 == 0 and self.score != 0:
                self.skill_flag = True
                sfx_thread = threading.Thread(target=play_selected_sfx, args=(sfx_3_path,))
                sfx_thread.start()

    # 먹이 그려주기
    def draw_Food(self, imgMain):
        rx, ry = self.foodPoint
        socketio.emit('foodPoint', {'food_x': rx, 'food_y': ry})
        imgMain = cvzone.overlayPNG(imgMain, self.imgFood, (rx - self.wFood // 2, ry - self.hFood // 2))

        return imgMain

    # 데이터 수신 (udp 통신 일때만 사용)
    def receive_data_from_opp(self):
        for _ in range(self.con_cnt):
            try:
                data, _ = self.sock.recvfrom(15000)
                decode_data = data.decode()
                self.queue.append(decode_data)
                self.udp_count = 0
                
            except socket.timeout:
                self.udp_count += 1
                if self.udp_count > 25:
                    socketio.emit('opponent_escaped')
            except BlockingIOError:
                self.udp_count += 1
                if self.udp_count > 40:
                    socketio.emit('opponent_escaped')

        if len(self.queue) == 0:
            pass
        elif len(self.queue) > 4:
            for _ in range(len(self.queue) // 4):
                self.queue.pop(0)
            temp = self.queue.pop(0)
            if temp[0] == '[':
                self.opp_points = eval(temp)
        else:
            temp = self.queue.pop(0)
            if temp[0] == '[':
                self.opp_points = eval(temp)
                
            
    def draw_triangle(self, point, point2, size):
        x,y=point
        x2,y2=point2
        triangle_size = size
        half_triangle_size = int(triangle_size / 2)
        
        triangle = [(0, 0 - half_triangle_size),(0 - half_triangle_size, 0 + half_triangle_size),(0 + half_triangle_size, 0 + half_triangle_size)]

        angle =  math.atan2(y2-y,x2-x) -90*math.pi/180
        r_m = [
                [math.cos(angle), -math.sin(angle)],
                [math.sin(angle), math.cos(angle)]
            ]
        rotated_triangle = [[int(vertex[0]*r_m[0][0]+vertex[1]*r_m[0][1]+x), int(vertex[0]*r_m[1][0]+vertex[1]*r_m[1][1]+y)] for vertex in triangle]
        triangle_pts = np.array(rotated_triangle, np.int32).reshape((-1,1,2))
        return triangle_pts
    
    # 뱀 그려주기
    def draw_snakes(self, imgMain, points, HandPoints, isMe):

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

        # --- head point와 hands point 이어주기 ---
        if isMe and HandPoints and self.line_flag:
            for p in np.linspace(self.previousHead, HandPoints, 10):
                cv2.circle(imgMain, tuple(np.int32(p)), 5, (0, 255, 0), -1)

        # --- skill flag에 따라 색 바꾸기 --- 
        skill_colored = False
        if isMe:
            skill_colored = self.skill_flag
        else:
            skill_colored = self.opp_skill_flag

        if skill_colored:
            cv2.polylines(imgMain, np.int32([pts]), False, rainbow, 15)

            triangle_pts=self.draw_triangle(points[-1][1],points[-1][0], 50)
            triangle_pts_back=self.draw_triangle(points[-1][1],points[-1][0], 35)
            # cv2.polylines(imgMain, np.int32([triangle_pts1]), False, rainbow, 15)
            # cv2.polylines(imgMain, np.int32([triangle_pts2]), False, rainbow, 15)
            cv2.fillPoly(imgMain, [triangle_pts], megenta)
            cv2.fillPoly(imgMain, [triangle_pts_back], rainbow)

        else:
            cv2.polylines(imgMain, np.int32([pts]), False, maincolor, 15)
            if points:
                cv2.circle(imgMain, points[-1][1], 20, bodercolor, cv2.FILLED)
                cv2.circle(imgMain, points[-1][1], 15, rainbow, cv2.FILLED)

        return imgMain

    # 데이터 전송
    def send_data_to_opp(self):
        if self.is_udp:
            data_set = str(self.points)
            try:
                self.sock.sendto(data_set.encode(), self.opp_addr)
            except socket.timeout:
                pass
            except BlockingIOError:
                pass
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
            return 0
        # p1_a, p1_b = np.array(u1_head_pt[0]), np.array(u1_head_pt[1]) # p1_b: head point
        p1_a, p1_b = u1_head_pt[0], u1_head_pt[1]

        for idx, u2_pt in enumerate(u2_pts):
            # p2_a, p2_b = np.array(u2_pt[0]), np.array(u2_pt[1])
            p2_a, p2_b = u2_pt[0], u2_pt[1]

            if self.segmentIntersects(p1_a, p1_b, p2_a, p2_b):
                # print(p1_a, p1_b, p2_a, p2_b)
                return idx

        return 0

    # skill 사용 시 충돌 idx 자르기
    def skill_length_reduction(self):
        for i in range(self.cut_idx):
            self.currentLength -= self.lengths[i]

        if self.currentLength < 100:
            self.allowedLength = 100
        else:
            self.allowedLength = self.currentLength

        self.lengths = self.lengths[self.cut_idx:]
        self.points = self.points[self.cut_idx:]

    # 뱀이 충돌했을때
    def execute(self):
        self.check_collision = False
        self.user_move = False
        self.gen = False
        # 상대에게 게임오버 플래그 보내기 전 슬립줘서 상대 화면에도 박은게 보이게 하는 Sleep
        # time.sleep(0.25)
        socketio.emit('gameover')

    # 소멸자 소켓 bind 해제
    def __del__(self):
        self.sock.close()


########################################################################################################################
######################################## FLASK APP ROUTINGS ############################################################

game = SnakeGameClass(pathFood)
multi = MultiGameClass(pathFood)
single_game = SnakeGameClass(pathFood)


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
    global single_game

    single_game = SnakeGameClass(pathFood)
    folder_path =os.path.join(current_dir, 'static')
    filename = "bestScore.txt"
    file_path = os.path.join(folder_path, filename)

    myBestScore = 0
    if os.path.isfile(file_path):  # check if file exists
        with open(file_path, "r") as f:
            myBestScore = int(f.read())
    else:  # create the file if it doesn't exist
        with open(file_path, "w") as f:
            f.write("0")

    single_game.bestScore = myBestScore
    # print(f"bestScore : {single_game.bestScore}")
    return render_template("testbed.html", best_score=single_game.bestScore)


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
    sid = multi.user_number

    # 시연 위해 UDP 연결 비활성화하고 바로 Game Ready Emit
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
    if multi.opp_score % 5 == 0 and multi.opp_score != 0:
        multi.opp_skill_flag = True
        sfx_thread = threading.Thread(target=play_selected_sfx, args=(sfx_6_path,))
        sfx_thread.start()
        socketio.emit('warning', {'opp_skill' : 1})

    multi.foodOnOff = True


# 게임 시작
@socketio.on('game_start')
def set_start():
    global start
    start = True


@socketio.on('cutted_idx')
def set_cutted_idx(data):
    global multi
    multi.cut_idx = data['cutted_idx']
    multi.skill_length_reduction()
    multi.opp_skill_flag=False


@socketio.on("save_best")
def save_best(data):
    with open(os.path.join(current_dir, 'static', 'bestScore.txt'), "w") as f:
        # Write the new contents to the file
        f.write(data)


@socketio.on("gen_break")
def gen_break():
    global multi
    multi.gen = False


########################################################################################################################
######################################## MAIN GAME ROUNTING ############################################################
@app.route('/snake')
def snake():
    def generate():
        global multi
        global start
        start = False
        skill_cnt = 0
        opp_skill_cnt = 0

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
            if start:
                break

        while True:
            success, img = cap.read()
            img = cv2.flip(img, 1)

            try:
                hands = detector.findHands(img, flipType=False)
                img = detector.drawHands(img)
            except:
                hands = []

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
                        multi.line_flag = True
                elif multi.user_number == 2:
                    start_cx -= 5
                    if start_cx < 930:
                        start_cx = 1210
                        multi.user_move = True
                        multi.check_collision = True
                        multi.line_flag = True

            if multi.skill_flag:
                skill_cnt += 1
                if skill_cnt % 120 == 0:
                    multi.skill_flag = False
                    skill_cnt = 0

            if multi.opp_skill_flag:
                opp_skill_cnt += 1
                if opp_skill_cnt % 120 == 0:
                    multi.opp_skill_flag = False
                    opp_skill_cnt = 0

            img = multi.update(img, pointIndex)

            # encode the image as a JPEG string
            _, img_encoded = cv2.imencode('.jpg', img)
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n')

            if not multi.gen:
                break

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


########################################################################################################################
############################### TEST BED FOR GAME LOGIC DEV ############################################################

# SETTING UP VARIABLES AND FUNCTION FOR BOT
bot_data = {'bot_head_x': 1000,
            'bot_head_y': 360,
            'bot_body_node': [],
            'currentLength': 0,
            'lengths': [],
            'bot_velocityX': random.choice([-1, 1]),
            'bot_velocityY': random.choice([-1, 1])}
bot_cnt = 0


def bot_data_update():
    global bot_data, bot_cnt

    bot_speed = 10
    px, py = bot_data['bot_head_x'], bot_data['bot_head_y']

    # 1초 마다 방향 바꾸기
    # print(bot_cnt)
    if bot_cnt == 30:
        bot_data['bot_velocityX'] = random.choice([-1, 0, 1])
        if bot_data['bot_velocityX'] == 0:
            bot_data['bot_velocityY'] = random.choice([-1, 1])
        else:
            bot_data['bot_velocityY'] = random.choice([-1, 0, 1])
        bot_cnt = 0
    bot_cnt += 1

    bot_velocityX = bot_data['bot_velocityX']
    bot_velocityY = bot_data['bot_velocityY']

    cx = round(px + bot_velocityX * bot_speed)
    cy = round(py + bot_velocityY * bot_speed)

    if cx < 0 or cx > 1280 or cy < 0 or cy > 720:
        if cx < 0: cx = 0
        if cx > 1280: cx = 1280
        if cy < 0: cy = 0
        if cy > 720: cy = 720

    if cx == 0 or cx == 1280:
        bot_data['bot_velocityX'] = -bot_data['bot_velocityX']
    if cy == 0 or cy == 720:
        bot_data['bot_velocityY'] = -bot_data['bot_velocityY']

    bot_data['bot_head_x'] = cx
    bot_data['bot_head_y'] = cy
    bot_data['bot_body_node'].append([[px, py], [cx, cy]])

    distance = math.hypot(cx - px, cy - py)
    bot_data['lengths'].append(distance)
    bot_data['currentLength'] += distance

    socketio.emit('bot_data', {'head_x': cx, 'head_y': cy})

    if bot_data['currentLength'] > 250:
        for i, length in enumerate(bot_data['lengths']):
            bot_data['currentLength'] -= length
            bot_data['lengths'] = bot_data['lengths'][1:]
            bot_data['bot_body_node'] = bot_data['bot_body_node'][1:]

            if bot_data['currentLength'] < 250:
                break


# TEST BED ROUTING
@app.route('/test')
def test():
    def generate():
        global bot_data, single_game, gameover_flag, bot_flag, user_move
        global opponent_data
        single_game.global_intialize()
        single_game.testbed_initialize()
        max_time_end = time.time() + 4
        cx, cy = 100, 360
        bot_flag = True
        user_move = False
        single_game.foodtimeLimit = time.time() + 15  # 10초 제한(앞 5초는 카운트)

        while True:
            success, img = cap.read()
            img = cv2.flip(img, 1)
            hands = detector.findHands(img, flipType=False)
            img = detector.drawHands(img)

            if not user_move:
                cx += 5
                pointIndex = [cx, cy]
            else:
                if hands:
                    lmList = hands[0]['lmList']
                    pointIndex = lmList[8][0:2]

            bot_data_update()
            opponent_data['opp_body_node'] = bot_data["bot_body_node"]
            # print(pointIndex)

            img = single_game.update(img, pointIndex)

            # encode the image as a JPEG string∂
            _, img_encoded = cv2.imencode('.jpg', img)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n')

            if time.time() > max_time_end:
                user_move = True
                single_game.line_flag = True

            remain_time = 0
            if user_move:
                remain_time = int(single_game.foodtimeLimit - time.time())  # 할일: html에 보내기
                # print(f"remain_time: {remain_time}")
                socketio.emit('test_timer', {"seconds": remain_time})

            if gameover_flag or (remain_time < 1 and user_move):
                print("game ended")
                gameover_flag = False
                socketio.emit('gameover')
                break

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Main Menu Selection
@app.route('/menu_snake')
def menu_snake():
    menu_game = SnakeGameClass(pathFood)

    menu_game.multi = False
    menu_game.foodOnOff = False
    menuimg = np.zeros((720, 1280, 3), dtype=np.uint8)
    menu_game.global_intialize()
    menu_game.menu_initialize()

    def generate():
        while True:
            success, img = cap.read()
            img = cv2.flip(img, 1)
            hands = detector.findHands(img, flipType=False)
            showimg = detector.drawHands(menuimg)
            pointIndex = []

            if hands:
                lmList = hands[0]['lmList']
                pointIndex = lmList[8][0:2]

            showimg = menu_game.update_blackbg(showimg, pointIndex)
            # encode the image as a JPEG string
            _, img_encoded = cv2.imencode('.jpg', showimg)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


def create_maze(image_h, image_w, block_rows, block_cols):
    manager = MazeManager()
    maze = manager.add_maze(block_rows, block_cols)
    manager.solve_maze(maze.id, "DepthFirstBacktracker")

    wall_map = np.zeros((image_h, image_w))  # (h,w)
    block_h = image_h // block_rows
    block_w = image_w // block_cols

    start = [[], []]
    end = [[], []]
    r = 2

    for i in range(block_rows):
        for j in range(block_cols):
            if maze.initial_grid[i][j].is_entry_exit == "entry":
                start = [[j * block_w + 150, i * block_h + 150], [(j + 1) * block_w + 150, (i + 1) * block_h + 150]]
                wall_map[i * block_h + 10: (i + 1) * block_h - 10, j * block_w + 10: (j + 1) * block_w - 10] = 2
                # print(f"start in create_maze: {start}")
            elif maze.initial_grid[i][j].is_entry_exit == "exit":
                end = [[j * block_w + 150, i * block_h + 150], [(j + 1) * block_w + 150, (i + 1) * block_h + 150]]
                wall_map[i * block_h + 10: (i + 1) * block_h - 10, j * block_w + 10: (j + 1) * block_w - 10] = 3
                # print(f"end in create_maze:{end}")
            if maze.initial_grid[i][j].walls["top"]:
                if i == 0:
                    wall_map[i * block_h:i * block_h + r, j * block_w:(j + 1) * block_w] = 1
                else:
                    wall_map[i * block_h - r:i * block_h + r, j * block_w:(j + 1) * block_w] = 1
            if maze.initial_grid[i][j].walls["right"]:
                wall_map[i * block_h:(i + 1) * block_h, (j + 1) * block_w - r:(j + 1) * block_w + r] = 1
            if maze.initial_grid[i][j].walls["bottom"]:
                wall_map[(i + 1) * block_h - r:(i + 1) * block_h + r, j * block_w:(j + 1) * block_w] = 1
            if maze.initial_grid[i][j].walls["left"]:
                if j == 0:
                    wall_map[i * block_h:(i + 1) * block_h, j * block_w:j * block_w + r] = 1
                else:
                    wall_map[i * block_h:(i + 1) * block_h, j * block_w - r:j * block_w + r] = 1

    solution_nodes = maze.solution_path
    mid_goal_h = maze.solution_path[-3][0][0]  # solution path의 출구로부터 2번쨰 노드
    mid_goal_w = maze.solution_path[-3][0][1]
    # print(len(solution_nodes))
    mid = [[mid_goal_w * block_w + 150, mid_goal_h * block_h + 150],
           [(mid_goal_w + 1) * block_w + 150, (mid_goal_h + 1) * block_h + 150]]
    # wall_map[mid_goal_h * block_h : (mid_goal_h + 1) * block_h , mid_goal_w * block_w :(mid_goal_w + 1) * block_w] = 4

    return start, mid, end, wall_map


@app.route('/maze_play')
def maze_play():
    def generate():
        global game

        game.multi = False
        game.maze_initialize()

        game.timer_end = time.time() + 91  # 1분 30초 시간제한

        while True:
            success, img = cap.read()
            img = cv2.flip(img, 1)

            hands = detector.findHands(img, flipType=False)
            showimg = detector.drawHands(game.maze_img)  # 무조건 findHands 다음

            pointIndex = []
            if hands:
                lmList = hands[0]['lmList']
                pointIndex = lmList[8][0:2]

            showimg = game.update_mazeVer(showimg, pointIndex)

            # encode the image as a JPEG string
            _, img_encoded = cv2.imencode('.jpg', showimg)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n')

            remain_time = int(game.timer_end - time.time())  # 할일: html에 보내기
            # print(f"remain_time: {remain_time}")
            # socketio.emit('maze_timer', {"minutes": remain_time // 60, "seconds": remain_time % 60})
            socketio.emit('maze_timer', {"remain_time": remain_time})
            if remain_time < 1:
                print("game ended")
                socketio.emit('gameover')
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
