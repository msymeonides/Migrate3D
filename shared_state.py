import threading

messages = []
thread_lock = threading.RLock()
progress_value = 0
_progress_steps = {}
_completed_steps = set()
_abort_flag = False

def set_progress(value):
    global progress_value
    with thread_lock:
        progress_value = min(max(0, value), 100)

def get_progress():
    with thread_lock:
        return progress_value

def init_progress_tracker(optional_flags):
    global _progress_steps, _completed_steps
    steps = ["Formatting", "Calculations", "MSD", "Summary", "Final results save"]
    if optional_flags.get("pca_xgb", False):
        steps.extend(["PCA", "XGB"])
    if optional_flags.get("contacts", False):
        steps.append("Contacts")
    if optional_flags.get("attractors", False):
        steps.append("Attractors")
    if optional_flags.get("generate_figures", False):
        steps.append("Generate Figures")
    weight = 100 / len(steps)
    _progress_steps = {step: weight for step in steps}
    _completed_steps = set()
    set_progress(0)

def complete_progress_step(step_name):
    global _completed_steps
    with thread_lock:
        if step_name in _completed_steps:
            return
        _completed_steps.add(step_name)
        _update_progress()
        if not _abort_flag:
            print(f"Completed {step_name}")

def _update_progress():
    total = sum(_progress_steps.get(step, 0) for step in _completed_steps)
    set_progress(total)

def set_abort_state():
    global _abort_flag
    with thread_lock:
        _abort_flag = True
        set_progress(100)

def is_aborted():
    with thread_lock:
        return _abort_flag
