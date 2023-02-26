a = {}

room_id = 'asdf'

if room_id not in a:
    print('no')
    a[room_id] = []
    a[room_id].append(1)

if room_id not in a:
    print("?")
else:
    print(a[room_id])