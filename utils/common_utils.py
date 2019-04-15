import os

def checkdir(_dir):
    if not os.path.exists(_dir):
        os.mkdir(_dir)
    return _dir