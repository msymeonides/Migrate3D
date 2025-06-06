import threading

messages = []
thread_lock = threading.Lock()
progress_value = 0

def get_progress():
    with thread_lock:
        return progress_value

def set_progress(value):
    global progress_value
    with thread_lock:
        progress_value = min(max(0, value), 100)