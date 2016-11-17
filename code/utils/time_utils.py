import datetime

def _timestamp():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d-%H-%M")

