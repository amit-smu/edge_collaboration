def test(l):
    l1 = l.copy()
    l1.extend(list(range(15, 20)))
    print(l1)


a = list(range(1, 10))
b = list(range(6, 10))

print("value of a initially :{}".format(a))
test(a)
print("value of a after :{}".format(a))
