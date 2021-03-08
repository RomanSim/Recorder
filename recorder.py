import time
import enum
import wave
import queue
import random
import logging
import datetime
import threading
import collections
from numpy import *
import dec_lib as spl
import numpy as np
from scipy.signal import lfilter
import pyaudio
import socket
import requests
import json
import base64
import pprint

SAMPLE_RATE_FRAMES_PER_SECOND = 44100
SAMPLE_RATE_SECONDS = 1
SAMPLE_RATE_TOTAL_FRAMES = SAMPLE_RATE_FRAMES_PER_SECOND*SAMPLE_RATE_SECONDS
SMALL_WINDOW_SECONDS_SIZE = 3
BIG_WINDOW_SECONDS_SIZE = 10
DUMP_WINDOW_LOOKAHEAD_IN_SECONDS = 5
RECORD_SECONDS_BACKGROUND = 10
CHUNK = 1024


class Recorder(object):
    pass


class Analyzer(object):
    pass


class AudioWindow(object):
    def __init__(self, *, frame_size_in_bytes, channels, rate, seconds):
        self._frame_size_in_bytes = frame_size_in_bytes
        self._channels = channels
        self._rate = rate

        self._bytes_per_second = self._frame_size_in_bytes * self._channels * self._rate
        self._total_window_bytes = self._bytes_per_second * int(seconds)

        self._data = b""

    def add(self, chunk):
        # Add the chunk to the data
        data = b"".join([self._data, chunk])

        # Trim to the window size
        data = data[-1*self._total_window_bytes:]

        self._data = data

    @property
    def data(self):
        return self._data

    @property
    def last_n(self, seconds):
        pass

    @property
    def seconds(self):
        return len(self._data) / float(self._bytes_per_second)

    @property
    def size(self):
        return len(self._data)


class BusType(enum.IntEnum):
    AUDIO = 0
    EVENT = 1
    TICK = 2

    QUIT = 1000


# def dB_meter(data):

#     NUMERATOR, DENOMINATOR = spl.A_weighting(SAMPLE_RATE_FRAMES_PER_SECOND)
#     decoded_block = np.frombuffer(data, 'Int16')
#     y = lfilter(NUMERATOR, DENOMINATOR, decoded_block)
#     new_decibel = 20 * np.log10(spl.rms_flat(y))
#     return new_decibel


def Pitch(signal):
    signal = np.fromstring(signal, 'Int16')
    crossing = [math.copysign(1.0, s) for s in signal]
    index = find(np.diff(crossing))
    f0 = round(len(index) * SAMPLE_RATE_FRAMES_PER_SECOND /
               (2 * np.prod(len(signal))))
    return f0


def find(condition):
    res, = np.nonzero(np.ravel(condition))
    return res


def peek_dB(data):
    check_dB = Pitch(data)
    return check_dB


def mostFrequent(arr):

    arr.sort()
    i = 0
    count = 0
    k = 0
    reduct_arr = []
    arr_length = len(arr)
    while k < int(len(arr)):
        if arr[i] == arr[k]:
            count = count + 1
            k = k + 1
            if count >= 10 and arr_length - k == 1:
                x = arr[i]
                reduct_arr.append(x)
                break
            # maybe if statement like if arr[k+1] != null then enter the second if elif just check for count maybe
            if (count >= 10 and (arr[i] != arr[k + 1])):
                x = arr[i]
                reduct_arr.append(x)
                count = 0
                i = k + 1
        else:
            count = 0
            if i == 0:
                i = i + k
            else:
                i = k

    return reduct_arr


def calibration_reader():
    p = pyaudio.PyAudio()
    first_stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE_FRAMES_PER_SECOND,
        input=True
    )

    frames = []
    for i in range(0, int(SAMPLE_RATE_FRAMES_PER_SECOND/CHUNK*RECORD_SECONDS_BACKGROUND)):
        dB_Background_noise = first_stream.read(CHUNK)
        # dB = dB_meter(dB_Background_noise)
        dB = Pitch(dB_Background_noise)
        frames.append(int(dB))
    n = len(frames)
    print(n)
    most_Frequent = mostFrequent(frames)
    boundsArray = np.array(most_Frequent)
    print(boundsArray)
    # x = np.amax(boundsArray)
    # print(x)
    first_stream.stop_stream()
    first_stream.close()
    p.terminate()
    # return x
    return boundsArray


def reader(outgoing, incoming, bounds):
    # Initialize the audio
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE_FRAMES_PER_SECOND,
        input=True
    )
    print("listening...")
    # As long as active
    while incoming.empty():
        # Read full buffer and pish
        data = stream.read(SAMPLE_RATE_TOTAL_FRAMES)

        # Add data to the window
        outgoing.put_nowait((BusType.AUDIO, data))

    stream.stop_stream()
    stream.close()
    audio.terminate()


def analyzer(outgoing, incoming, bounds):
    # Initialize the window
    window = AudioWindow(
        frame_size_in_bytes=2,
        channels=1,
        rate=SAMPLE_RATE_FRAMES_PER_SECOND,
        seconds=SMALL_WINDOW_SECONDS_SIZE
    )
    lastRecord = 0
    # Process incoming packets
    while True:
        data = incoming.get()
        some_data = data[0]
        dB_vol = peek_dB(some_data)
        # Handle stop
        if data is None:
            return

        if not data or not isinstance(data[0], bytes):
            continue

        # Add to the window
        logging.info("Analyzer received bytes %d", len(data[0]))
        window.add(data[0])

        # Analyze event
        tryRecord = time.time()
        timeGap = tryRecord - lastRecord
        print(dB_vol)
        flag = False

        for i in range(len(bounds)):
            if dB_vol == bounds[i]:
                flag = False
                print(dB_vol, bounds[i])
                i = len(bounds)
            else:
                flag = True
        # if dB_vol > bounds and timeGap > 10:
        if flag and timeGap > 10:
            print("recording...")
            lastRecord = time.time()
            outgoing.put_nowait((BusType.EVENT, time.time()))


def dumper(outgoing, incoming, bounds):
    # Initialize the window
    window = AudioWindow(
        frame_size_in_bytes=2,
        channels=1,
        rate=SAMPLE_RATE_FRAMES_PER_SECOND,
        seconds=BIG_WINDOW_SECONDS_SIZE
    )

    # Process incoming packets
    while True:
        data = incoming.get()

        # Handle stop
        if data is None:
            return

        # Handle window data
        if data and isinstance(data[0], bytes):
            logging.info("Dumper received bytes %d", len(data[0]))
            window.add(data[0])

        # Handle dump
        elif data and isinstance(data[0], str):
            logging.info("Dumper received dump request %s", data[0])

            # Open the file with the correct configuration
            output = wave.open(data[0], "wb")
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(SAMPLE_RATE_FRAMES_PER_SECOND)
            output.writeframes(window.data)
            # url = 'http://127.0.0.1:8000/api/audio'

            # payload = {}
            # file_name = data[0]
            # payload['data'] = window.data.decode("iso-8859-1")
            # payload['name'] = file_name

            # try:
            #     r = requests.post(url, json=payload)
            #     print(r.status_code)
            # except:
            #     logging.info("something wrong")


def scheduler(outgoing, incoming, bounds):
    schedules = collections.deque()

    # Process incoming packets
    while True:
        # Calculate the timeout to the next schedule
        try:
            # Calculate when is the next timeout
            next_timeout = None
            if schedules:
                next_timeout = max(0, schedules[0]-time.time())
                if not next_timeout:
                    # Fake timeout to give priority to the tick
                    raise queue.Empty()

            data = incoming.get(timeout=next_timeout)
        except queue.Empty:
            outgoing.put_nowait((BusType.TICK, None))
            schedules.popleft()
            continue

        # Handle stop
        if data is None:
            return

        # Add to the schedulers
        if data:
            logging.info("Scheduler will tick in %d seconds", data[0])
            schedules.append(time.time() + data[0])


class Task(object):
    def __init__(self, bus, task, int):
        self._bus = bus
        self._task = task
        self._thread = None
        self._queue = queue.Queue()
        self._bounds = bounds
        self._thread = threading.Thread(
            target=self._worker
        )

    def start(self):
        logging.info("Task started %s", self)
        self._thread.start()

    def stop(self):
        logging.info("Task stopping %s", self)
        self._queue.put_nowait(None)

    def join(self):
        self._thread.join()

    def send(self, *args):
        self._queue.put_nowait(args)

    def _worker(self):
        try:
            self._task(self._bus, self._queue, self._bounds)
        finally:
            self._bus.put_nowait((BusType.QUIT, self))

    def __str__(self):
        return f"<{self._task.__name__}>"


def format_now_filename():
    ts = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return f"dump_{ts}.wav"


if "__main__" == __name__:
    logging.basicConfig(level=logging.DEBUG)

    # Initialize bus and tasks
    bounds = calibration_reader()
    bus = queue.Queue()
    task_reader = Task(bus, reader, bounds)
    task_analyzer = Task(bus, analyzer, bounds)
    task_dumper = Task(bus, dumper, bounds)
    task_scheduler = Task(bus, scheduler, bounds)

    tasks = [
        task_reader,
        task_analyzer,
        task_dumper,
        task_scheduler
    ]

    # Start tasks
    for task in tasks:
        task.start()

    # Process bus
    try:
        while True:
            # Get a single bus message
            message, data = bus.get()

            logging.info("Reactor received event %s", message)

            # Handle messages
            if message == BusType.AUDIO:
                task_analyzer.send(data)
                task_dumper.send(data)

            elif message == BusType.EVENT:
                task_scheduler.send(
                    DUMP_WINDOW_LOOKAHEAD_IN_SECONDS
                )

            elif message == BusType.TICK:
                task_dumper.send(format_now_filename())

            elif message == BusType.QUIT:
                break
    except KeyboardInterrupt:
        pass

    # Send stop to all
    for task in tasks:
        task.stop()

    # Join tasks
    for task in tasks:
        task.join()
