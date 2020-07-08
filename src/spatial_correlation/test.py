l = []
for i in range(5):
    d = {
        'key1': i,
        'key2': i + 10
    }
    l.append(d)

print(l)
for d in l:
    if d['key1'] == 2:
        l.remove(d)
        print(l)

print(l)
