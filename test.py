a = ["happy", "happy", "sad", "angry"]
l = len(a)
new = {e: f"{a.count(e) / l * 100:.2f} %" for e in a}
print(new)
