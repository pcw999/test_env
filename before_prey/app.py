# =========== module import ===========
import time
import math
import random
import cvzone
import sys
import os
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, Response, request, redirect, url_for, session
from flask_socketio import SocketIO, emit, join_room, leave_room
from engineio.payload import Payload
import socket
# ======================================

# =========== Program config setting ===========
Payload.max_decode_packets = 200

print(f"flask is running in {os.getcwd()}, __name__ is {__name__}", flush=True)
print(sys.version, flush=True)

app = Flask(__name__)
app.config['SECRET_KEY'] = "roomfitisdead"

socketio = SocketIO(app, cors_allowed_origins='*')
# ================================================

# =========== HandDetector ===========
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
# ====================================

# =========== Global variables ===========
opponent_data = {} # 상대 데이터 (현재 손위치, 현재 뱀위치)
gameover_flag = False # ^^ 게임오버
now_my_room = "" # 현재 내가 있는 방
now_my_sid = "" # 현재 나의 sid
MY_PORT = 0 # socket_bind를 위한 내 포트 번호
# ====================================

############################## SNAKE GAME LOGIC SECTION ##############################
# video setting
cap = cv2.VideoCapture(0)
# Ubuntu YUYV cam setting low frame rate problem fixed
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
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
    def __init__(self, pathFood):
        self.points = []
        self.lengths = []
        self.currentLength = 0
        self.allowedLength = 150
        self.previousHead = 0, 0
        self.score = 0
        self.opp_score = 0
        self.opp_bodys = []

        self.speed = 5
        self.minspeed=10
        self.maxspeed=math.hypot(1280, 720) / 10
        self.velocityX = random.choice([-1, 0, 1])
        self.velocityY = random.choice([-1, 1])

        self.imgFood = cv2.imread(pathFood, cv2.IMREAD_UNCHANGED)
        self.hFood, self.wFood, _ = self.imgFood.shape
        self.foodPoint = 640, 360
        self.foodOnOff = True

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.opp_addr = ()
        self.is_udp = False
        self.udp_count = 0

        self.gameover = False

    def ccw(self, p, a, b):
        s = p[0] * a[1] + a[0] * b[1] + b[0] * p[1]
        s -= (p[1] * a[0] + a[1] * b[0] + b[1] * p[0])

        if s > 0 :
            return 1
        elif s == 0 :
            return 0
        else :
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
        p1_a, p1_b = u1_head_pt[0], u1_head_pt[1]

        for u2_pt in u2_pts:
            p2_a, p2_b = u2_pt[0], u2_pt[1]
            if self.segmentIntersects(p1_a, p1_b, p2_a, p2_b):
                return True
        return False

    def draw_snakes(self, imgMain, points, isMe):
        bodercolor = cyan
        maincolor = red

        if isMe:
            bodercolor = megenta
            maincolor = green

        # Change hue every 100ms
        change_interval = 100

        hue = int(time.time() * change_interval % 180) # TODO : 마지막에 성능 부족 시 아낄 수 있음
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

    def draw_Food(self, imgMain):
        rx, ry = self.foodPoint
        socketio.emit('foodPoint', {'food_x': rx, 'food_y': ry})
        imgMain = cvzone.overlayPNG(imgMain, self.imgFood, (rx - self.wFood // 2, ry - self.hFood // 2))

        return imgMain

    def my_snake_update(self, HandPoints, opp_bodys):
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

        self.send_data_to_opp()
        self.send_data_to_html()

        if self.is_udp:
            self.receive_data_from_opp()

        if len(self.points) != 0: #out of range 용 성능 애바면 좀;;
            if self.isCollision(self.points[-1], opp_bodys):
                print("hit")
                self.execute()
        else:
            print('point가 텅텅 !')

    def set_snake_speed(self, HandPoints, s_speed):
        px, py = self.previousHead
        # ----HandsPoint moving ----
        s_speed = 20
        if HandPoints:
            m_x, m_y = HandPoints
            dx = m_x - px  # -1~1
            dy = m_y - py

            # speed 범위: 0~1460
            if math.hypot(dx, dy) > self.maxspeed : # 146
                self.speed = self.maxspeed
            elif math.hypot(dx, dy) < self.minspeed:
                self.speed = self.minspeed
            else:
                self.speed = math.hypot(dx, dy)

            if dx != 0:
                self.velocityX = dx / 1280
            if dy != 0:
                self.velocityY = dy / 720

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

    def length_reduction(self):
        if self.currentLength > self.allowedLength:
            for i, length in enumerate(self.lengths):
                self.currentLength -= length
                self.lengths = self.lengths[1:]
                self.points = self.points[1:]

                if self.currentLength < self.allowedLength:
                    break

    def check_snake_eating(self, cx, cy):
        rx, ry = self.foodPoint
        if (rx - (self.wFood // 2) < cx < rx + (self.wFood // 2)) and (
                ry - (self.hFood // 2) < cy < ry + (self.hFood // 2)):
            self.allowedLength += 50
            self.score += 1

            self.foodOnOff = False
            socketio.emit('user_ate_food', {'score': self.score})

    def execute(self):
        self.points = []  # all points of the snake
        self.lengths = []  # distance between each point
        self.currentLength = 0  # total length of the snake
        self.allowedLength = 150  # total allowed Length

        self.gameover = True
        socketio.emit('gameover')

    def update(self, imgMain, HandPoints):
        global opponent_data

        # 0 이면 상대 뱀
        if opponent_data:
            self.opp_bodys = opponent_data['opp_body_node']
        imgMain = self.draw_snakes(imgMain, self.opp_bodys, 0)

        # update and draw own snake
        self.my_snake_update(HandPoints, self.opp_bodys)
        imgMain = self.draw_Food(imgMain)
        # 1 이면 내 뱀
        imgMain = self.draw_snakes(imgMain, self.points, 1)

        return imgMain

    def set_socket(self, my_port, opp_ip, opp_port):
        self.sock.bind(('0.0.0.0', int(my_port)))
        self.sock.settimeout(0.02) # TODO 만약 udp, 서버 선택 오류 시 다시 0.02로
        self.opp_addr = (opp_ip, int(opp_port))

    def send_data_to_opp(self):
        if self.is_udp:
            data_set = str(self.points)
            self.sock.sendto(data_set.encode(), self.opp_addr)
        else:
            socketio.emit('game_data', {'body_node': self.points})

    def send_data_to_html(self):
        socketio.emit('game_data_for_debug', {'score': self.score, 'fps': fps})

    def receive_data_from_opp(self):
        global opponent_data

        try:
            data, _ = self.sock.recvfrom(15000)
            decode_data = data.decode()
            if decode_data[0] == '[':
                opponent_data['opp_body_node'] = eval(decode_data)
                self.udp_count = 0
            else:
                test_code = decode_data
                self.sock.sendto(test_code.encode(), self.opp_addr)
        except socket.timeout:
            self.udp_count += 1
            if self.udp_count > 25:
                socketio.emit('opponent_escaped')

    def test_connect(self, sid):
        a = 0
        b = 0
        test_code = str(sid)

        for i in range(60):
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

        if a != 60 and b != 0:
            self.is_udp = True

        print(f"connection MODE : {self.is_udp} / a = {a}, b = {b}")
        socketio.emit('NetworkMode', {'UDP': self.is_udp})

    def __del__(self):
        global opponent_data
        opponent_data = {}
        self.sock.close()


######################################################################################

# 게임 전역 변수로 선언
game = SnakeGameClass("./static/food.png") # ^^ 이식 시 파일 경로 설정

# 로컬 flask에서 index.html 로딩
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# 로컬 flask에서 snake.html 로딩
@app.route("/enter_snake", methods=["GET"])
def enter_snake():
    global now_my_sid
    global now_my_room
    global game

    now_my_room = request.args.get('room_id')
    now_my_sid = request.args.get('sid')
    print(now_my_room, now_my_sid)

    game = SnakeGameClass("./static/food.png") # ^^ 이식 시 파일 경로 설정

    return render_template("snake.html", room_id=now_my_room, sid=now_my_sid)

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
    global game
    opp_ip = data['ip_addr']
    opp_port = data['port']
    sid = data['sid']

    game.set_socket(MY_PORT, opp_ip, opp_port)
    game.test_connect(sid)

# socketio로 받은 상대방 정보
@socketio.on('opp_data_transfer')
def opp_data_transfer(data):
    global opponent_data
    opponent_data = data['data']

# socketio로 받은 먹이 위치
@socketio.on('set_food_location')
def set_food_loc(data):
    global game
    game.foodPoint = data['foodPoint']
    game.foodOnOff = True

# socketio로 받은 먹이 위치와 상대 점수
@socketio.on('set_food_location_score')
def set_food_loc(data):
    global game
    game.foodPoint = data['foodPoint']
    game.opp_score = data['opp_score']
    game.foodOnOff = True

# snake 페이지에서 필요한 영상 전송
@app.route('/snake')
def snake():
    def generate():
        global opponent_data
        global game
        global gameover_flag

        while True:
            success, img = cap.read()
            img = cv2.flip(img, 1)
            hands, img = detector.findHands(img, flipType=False)

            pointIndex = []

            if hands:
                lmList = hands[0]['lmList']
                pointIndex = lmList[8][0:2]

            img = game.update(img, pointIndex)

            # encode the image as a JPEG string
            _, img_encoded = cv2.imencode('.jpg', img)
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n')

            if gameover_flag: # ^^ 게임 오버 시
                pass

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    socketio.run(app, host='localhost', port=5000)
