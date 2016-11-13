import sys

def LOG(path, text):
    with open(path, "a") as f:
        f.write(text)
