from copyreg import pickle
import datetime
import time
from flask import Flask, render_template, Response, request, redirect, url_for, session
from flask_socketio import SocketIO, emit, join_room
import uuid
from engineio.payload import Payload
from socket import *
import random

Payload.max_decode_packets = 200

# flask setting
app = Flask(__name__)
app.config['SECRET_KEY'] = "roomfitisdead"

# CORS setting
socketio = SocketIO(app, cors_allowed_origins='*')

# Global variables
waiting_user_idx = -1 # 현재 유저 인덱스
waiting_players = [] # 매칭 잡은 유저 목록
room_of_players = {} # 해당 sid에 할당된 룸
players_in_room = {} # 해당 room에 존재하는 sid들
address = {} # 먼저 방에 들어온 사람 주소 (전송을 위해 저장)
 
# server test page
@app.route("/")
def index():
    return render_template("servertime.html")

# socketio로 서버가 웹페이지와 연결된 경우
@socketio.on('connect')
def test_connect():
    ip_addr = request.remote_addr
    port = request.environ['REMOTE_PORT']
    print(f'Client connected: {ip_addr}:{port}')

# socketio로 서버가 웹페이지와 연결해제된 경우 players_in_room에서 해당 sid 제거
@socketio.on('disconnect')
def test_disconnect():
    global room_of_players
    global players_in_room
    
    room_id = room_of_players[request.sid]

    players_in_room[room_id].remove(request.sid)

    ip_addr = request.remote_addr
    port = request.environ['REMOTE_PORT']
    print(f'Client disconnected: {ip_addr}:{port}')

# index page에서 join 버튼을 눌렀을때
@socketio.on('join')
def handle_join():
    global waiting_players
    global waiting_user_idx
    global room_of_players

    if request.sid in waiting_players:
        print("이미 매칭 중인 유저입니다.")
        return

    waiting_players.append(request.sid)
    waiting_user_idx += 1

    if waiting_user_idx % 2 == 1:
        user1 = waiting_players[waiting_user_idx-1]
        user2 = waiting_players[waiting_user_idx]
        # 룸 id 할당
        room_id = str(uuid.uuid4())
        # sid에게 들어갈 방 알려줌

        # 매칭 잡힌 사실 index 페이지에 보내줌
        emit('matched', {'room_id' : room_id, 'sid' : user1}, to=user1)
        emit('matched', {'room_id' : room_id, 'sid' : user2}, to=user2)
        # 매칭완료
        emit('start-game', {'room_id' : room_id, 'sid' : user1}, to=user1)
        emit('start-game', {'room_id' : room_id, 'sid' : user2}, to=user2)
    else:
        emit('waiting', {'sid' : request.sid}, to=request.sid)

# 서버가 상대의 위치 전송    
@socketio.on('send_data')
def send_data(data):
    global players_in_room
    
    body_node = data['body_node']
    room_id = data['room_id']

    if len(players_in_room[room_id]) == 1:
        emit('opponent_escaped', to=request.sid)
    else:
        emit('opp_data', {'opp_body_node' : body_node, 'opp_room_id' : room_id}, broadcast=True, include_self=False, room=room_id)

# 서버와 통신 테스트용
@socketio.on('get_time')
def get_time():
    while True:
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        socketio.emit('time', {'time': current_time})
        socketio.sleep(1)

# 서버에서 현재 자신의 포트 받아오기
# < sock.bind() 작업에서 포트 번호 지정을 위해 필요 >
# < index -> snake 페이지 변환하면서 포트 변경됨 > => 매칭 시 그때 포트를 받은 후 연결 
@socketio.on('my_port')
def my_port(data):
    global players_in_room
    global address
    ip_addr = request.remote_addr
    port = request.environ['REMOTE_PORT']
    room_id = data['room_id']
    
    join_room(room_id)

    if room_id not in players_in_room: #만약 player_in_room['room_id']가 존재하지 않는다면 해당 룸에 유저가 아직 X
        players_in_room[room_id] = []
        players_in_room[room_id].append(request.sid)
    else:
        players_in_room[room_id].append(request.sid)

    emit('my_port', {'my_port':port})

    if len(players_in_room[room_id]) == 2:
        room_of_players[request.sid] = room_id
        emit('opponent_address', {'ip_addr' : ip_addr, 'port' : port}, broadcast=True, include_self=False, room=room_id)
        emit('opponent_address', {'ip_addr' : address[room_id][0], 'port' : address[room_id][1]}, to=request.sid)
    else:
        room_of_players[request.sid] = room_id
        address[room_id] = [ip_addr, port]

# 각 클라이언트에게 음식 좌표와 상대 점수 전송
@socketio.on('user_ate_food')
def provide_food_data(data):
    foodPoint=(random.randint(100, 1000), random.randint(100, 600))
    emit('ate_user', {'foodPoint':foodPoint}, to=request.sid)
    emit('ate_user_opp', {'foodPoint':foodPoint, 'opp_score':data['score']}, room=data['room_id'], include_self=False)

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=8080)
