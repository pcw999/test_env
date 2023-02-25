from copyreg import pickle
import datetime
import time
from flask import Flask, render_template, Response, request, redirect, url_for, session
from flask_socketio import SocketIO, emit, join_room
import uuid
from engineio.payload import Payload
from socket import *

Payload.max_decode_packets = 200

app = Flask(__name__)
app.config['SECRET_KEY'] = "roomfitisdead"

socketio = SocketIO(app, cors_allowed_origins='*')

waiting_players = []
room_of_players = {}
players_in_room = {}
last_created_room = ""

@app.route("/")
def index():
    return render_template("servertime.html")

@socketio.on('connect')
def test_connect():
    print('Client connected')

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')

@socketio.on('server_disconnect')
def room_disconnect(data):
    room_id = data['room_id']
    print(f'room_id = {room_id}')
    sid = data['sid']
    global room_of_players
    room_of_players = {k: v for k, v in room_of_players.items() if v != room_id}
    print('user left room')

@socketio.on('gameover_to_server')
def gameover_to_server(data):
    sid = data['sid']

    emit("gameover_to_clients", {'sid' : sid}, broadcast=True, include_self=False)
    print('gameover to clients')

@socketio.on('join')
def handle_join():
    global last_created_room
    if len(waiting_players) == 0:
        waiting_players.append(request.sid)
        last_created_room = str(uuid.uuid4())

        join_room(last_created_room)
        room_of_players[request.sid] = last_created_room
        emit('waiting', {'room_id' : last_created_room, 'sid' : request.sid}, to=last_created_room)
    else:
        host_sid = waiting_players.pop()
        room_id = room_of_players[host_sid]
        join_room(room_id)

        room_of_players[request.sid] = room_id

        last_created_room = ""
        print(room_of_players)
        emit('matched', {'room_id' : room_id, 'sid' : request.sid}, to=room_id)
        emit('start-game', {'room_id' : room_id, 'sid' : request.sid}, to=request.sid)
        emit('start-game', {'room_id' : room_id, 'sid' : host_sid}, to=host_sid)

@socketio.on('send_data')
def send_data(data):
    head_x = data['head_x']
    head_y = data['head_y']
    body_node = data['body_node']
    score = data['score']
    room_id = data['room_id']
    sid = data['sid']

    # print(head_x, head_y, score, room_id, sid)
    # print(f'head_x, head_y, score, room_id, sid')
    emit('opp_data', {'opp_head_x' : head_x, 'opp_head_y' : head_y, 'opp_body_node' : body_node, 'opp_score' : score, 'opp_room_id' : room_id, 'opp_sid' : sid}, broadcast=True, include_self=False)
    # emit('opp_data', {'opp_head_x' : head_x, 'opp_head_y' : head_y, 'opp_body_node' : body_node, 'opp_score' : score, 'opp_room_id' : room_id, 'opp_sid' : sid}, broadcast=True)

@socketio.on('send_data_bot')
def send_data_bot(data):
    head_x = data['head_x']
    head_y = data['head_y']
    body_node = data['body_node']
    score = data['score']
    room_id = data['room_id']
    sid = data['sid']

    # print(head_x, head_y, score, room_id, sid)
    emit('bot_data', {'bot_head_x' : head_x, 'bot_head_y' : head_y, 'bot_body_node' : body_node, 'bot_score' : score, 'bot_room_id' : room_id}, broadcast=True, include_self=False)
    # emit('opp_data', {'opp_head_x' : head_x, 'opp_head_y' : head_y, 'opp_body_node' : body_node, 'opp_score' : score, 'opp_room_id' : room_id, 'opp_sid' : sid}, broadcast=True)


@socketio.on('get_time')
def get_time():
    while True:
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        socketio.emit('time', {'time': current_time})
        socketio.sleep(1)


if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=8080, debug=True)
