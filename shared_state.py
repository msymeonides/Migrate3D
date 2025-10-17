from datetime import datetime
import threading


class TimestampedMessageList(list):
    def __init__(self):
        super().__init__()
        self.runtime_log = []

    def append(self, message):
        super().append(message)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.runtime_log.append({"Timestamp": timestamp, "Message": message})

    def __setitem__(self, index, value):
        super().__setitem__(index, value)
        if -len(self) <= index < len(self):
            if index < 0:
                index = len(self) + index
            while len(self.runtime_log) <= index:
                self.runtime_log.append({"Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Message": ""})

            if 0 <= index < len(self.runtime_log):
                self.runtime_log[index]["Message"] = value
                self.runtime_log[index]["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def get_runtime_log(self):
        return self.runtime_log.copy()

    def clear_runtime_log(self):
        self.runtime_log.clear()


messages = TimestampedMessageList()
thread_lock = threading.RLock()
progress_value = 0
progress_steps = {}
completed_steps = set()
abort_flag = False


def set_progress(value):
    global progress_value
    with thread_lock:
        progress_value = min(max(0, value), 100)


def get_progress():
    with thread_lock:
        return progress_value


def init_progress_tracker(optional_flags):
    global progress_steps, completed_steps
    steps = ["Formatting", "Calculations", "MSD"]
    if optional_flags.get("helicity", False):
        steps.append("Helicity")
    steps.append("Summary")
    if optional_flags.get("pca_xgb", False):
        steps.extend(["PCA", "XGB"])
    if optional_flags.get("contacts", False):
        steps.append("Contacts")
    if optional_flags.get("attractors", False):
        steps.append("Attractors")
    if optional_flags.get("generate_figures", False):
        steps.append("Generate Figures")
    steps.append("Final results save")
    weight = 100 / len(steps)
    progress_steps = {step: weight for step in steps}
    completed_steps = set()
    set_progress(0)


def complete_progress_step(step_name):
    global completed_steps
    with thread_lock:
        if step_name in completed_steps:
            return
        completed_steps.add(step_name)
        update_progress()
        if not abort_flag:
            print(f"Completed {step_name}")


def update_progress():
    total = sum(progress_steps.get(step, 0) for step in completed_steps)
    set_progress(total)


def set_abort_state():
    global abort_flag
    with thread_lock:
        abort_flag = True
        set_progress(100)


def is_aborted():
    with thread_lock:
        return abort_flag
