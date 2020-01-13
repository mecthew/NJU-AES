import time
import numpy as np
from typing import Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from modeling.constant import SCALE_SCORE

nesting_level = 0

asap_ranges = {
    1: (2.0, 12.0),
    2: (1.0, 6.0),
    3: (0.0, 3.0),
    4: (0.0, 3.0),
    5: (0.0, 4.0),
    6: (0.0, 4.0),
    7: (0.0, 30.0),
    8: (0.0, 60.0)
}

prompt_target = {
    1: [2],
    2: [1],
    3: [1, 4],
    4: [1, 3],
    5: [6],
    6: [5],
    7: range(1, 9),
    8: range(1, 9)
}


def normalize_score(y_origin, prompt_num, score_range=SCALE_SCORE):
    # min_origin = np.min(y_origin)
    # max_origin = np.max(y_origin)
    # min_origin = MIN_ORIGIN[prompt_num]
    # max_origin = MAX_ORIGIN[prompt_num]

    min_target = 0
    y_target = []
    for y_i in y_origin:
        min_origin, max_origin = asap_ranges.get(prompt_num)
        y_target_i = (y_i - min_origin) * 1.0 / (max_origin - min_origin) * score_range
        y_target.append(y_target_i)
    return np.array(y_target)


def inverse_score(y_origin, prompt_num, score_range=1.0):
    # min_origin = np.min(y_origin)
    # max_origin = np.max(y_origin)
    # min_origin = MIN_ORIGIN[prompt_num]
   #  max_origin = MAX_ORIGIN[prompt_num]

    score_range = SCALE_SCORE
    min_target = 0
    y_target = []
    for y_i in y_origin:
        min_origin, max_origin = asap_ranges.get(prompt_num)
        y_target_i = y_i * (max_origin - min_origin) * 1.0 / score_range + min_origin
        y_target.append(y_target_i)
    return np.array(y_target)


def log(entry: Any):
    global nesting_level
    space = "-" * (4 * nesting_level)
    print(f"{space}{entry}")


def log_prompt(entry: Any, prompt):
    global nesting_level
    space = "-" * (4 * nesting_level)
    delim = "="*10
    prompt_str = "@prompt" + str(prompt)
    print(f"{space}{delim} {entry}{prompt_str} {delim}")


def timeit(method, start_log=None):
    def wrapper(*args, **kw):
        global nesting_level

        log(f"Start [{method.__name__}]:" + (start_log if start_log else ""))
        nesting_level += 1

        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()

        nesting_level -= 1
        log(f"End   [{method.__name__}]. Time elapsed: {end_time - start_time:0.2f} sec.")
        return result

    return wrapper


def label_scalar(y):
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(y)
    y_scaler = min_max_scaler.transform(y)
    return min_max_scaler, y_scaler