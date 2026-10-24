"""Speaker separation runs in child processes, so one long recording cannot freeze the server.

sherpa-onnx holds Python's global lock for a whole call. The models here are stand-ins that do the
same on purpose (a C call that never lets go), which is what makes these tests mean something: the
in-process control must stall, and the isolated run must not.
"""

import ctypes
import multiprocessing
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from scribe_server.audio import RATE
from scribe_server.diarize import SHARE_FROM, Cancelled, Diarizer, SpeakerBook

HOLD_S = 1.5


class Models:
    """Stand-in for `Sherpa`. The first sample picks the behaviour: 1 holds the global lock for
    HOLD_S, 2 sleeps (lock free), 9 kills the process; anything else answers at once."""

    def __init__(self, role=None):
        pass

    def turns(self, audio, live, threshold=None):
        flag = int(audio[0])
        if flag == 1:
            ctypes.PyDLL(None).usleep(int(HOLD_S * 1e6))  # a C call that keeps the lock, as sherpa's do
        elif flag == 2:
            time.sleep(HOLD_S)
        elif flag == 9:
            os._exit(1)
        # One 2 s turn whose label carries what the process saw, to check the audio arrived whole.
        return [(0.0, 2.0, int(audio.sum()) % 1000), (3.0, 5.0, int(live) + 1000)]

    def voice(self, audio):
        return np.array([os.nice(0), len(audio)], dtype=np.float32)


def make(isolated=True):
    return Diarizer(Path("unused"), isolated=isolated, factory=Models)


def audio(flag, seconds=6):
    out = np.zeros(seconds * RATE, dtype=np.float32)
    out[0] = flag
    return out


class Ticker:
    """The longest stretch in which a thread of this process got no time: what a request would feel."""

    def __enter__(self):
        self.gaps = [0.0]
        self.stop = False

        def run():
            last = time.perf_counter()
            while not self.stop:
                time.sleep(0.01)
                now = time.perf_counter()
                self.gaps.append(now - last)
                last = now

        self.thread = threading.Thread(target=run)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop = True
        self.thread.join()

    @property
    def worst(self) -> float:
        return max(self.gaps)


def test_control_in_process_a_lock_holding_call_freezes_everything():
    d = make(isolated=False)
    with Ticker() as t:
        d.whole(audio(1))
    assert t.worst >= HOLD_S * 0.9  # the bug these tests guard against


def test_isolated_a_lock_holding_recording_does_not_freeze_this_process():
    d = make()
    try:
        with Ticker() as t:
            d.whole(audio(1))
        assert t.worst < 0.5
    finally:
        d.close()


def test_isolated_a_call_is_not_held_up_by_a_long_recording():
    d = make()
    try:
        done = threading.Thread(target=d.whole, args=(audio(2),))
        done.start()
        time.sleep(0.3)  # the recording is under way
        started = time.perf_counter()
        d.turns(audio(0), SpeakerBook())
        assert time.perf_counter() - started < 1.0  # it did not wait for the recording's 1.5 s
        done.join()
    finally:
        d.close()


def test_isolated_gives_the_same_turns_as_in_process():
    clip = audio(0)
    clip[100:200] = 1.0
    local, isolated = make(isolated=False), make()
    try:
        want = [(t.start, t.end, t.label) for t, _ in local.whole(clip)]
        got = [(t.start, t.end, t.label) for t, _ in isolated.whole(clip)]
        assert got == want and got
        assert [(t.start, t.end) for t, _ in isolated.whole(clip)] == [(0.0, 2.0), (3.0, 5.0)]
    finally:
        isolated.close()


def test_a_long_recording_goes_through_shared_memory_whole_and_leaves_none_behind():
    long = np.ones(SHARE_FROM + RATE, dtype=np.float32) * 0.001
    long[0] = 0
    blocks = lambda: {n for n in os.listdir("/dev/shm") if n.startswith("psm_")}  # noqa: E731
    before = blocks()
    d = make()
    try:
        (turn, _), *_ = d.whole(long)
        assert turn.label == int(long.sum()) % 1000  # the child saw every sample
        assert blocks() - before == set()
    finally:
        d.close()


def test_recordings_run_at_a_lower_priority_than_the_server_and_calls_do_not():
    d = make()
    try:
        assert d.voice(audio(0), live=False)[0] == min(os.nice(0) + 10, 19)
        assert d.voice(audio(0), live=True)[0] == os.nice(0)
    finally:
        d.close()


def test_a_dead_process_fails_that_call_and_the_next_one_works():
    d = make()
    try:
        with pytest.raises(RuntimeError, match="died"):
            d.whole(audio(9))
        assert d.whole(audio(0))  # a new process took over
    finally:
        d.close()


def test_restarting_a_dead_process_leaves_no_extra_processes():
    d = make()
    d.whole(audio(0))
    for _ in range(3):
        with pytest.raises(RuntimeError):
            d.whole(audio(9))
        d.whole(audio(0))
    assert len(multiprocessing.active_children()) == 2  # one per role, never the dead ones too
    d.close()
    assert multiprocessing.active_children() == []


def alive(*pids) -> list[int]:
    out = []
    for pid in pids:
        try:
            os.kill(pid, 0)
            if "Z" not in Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1][:1]:
                out.append(pid)
        except (ProcessLookupError, FileNotFoundError):
            pass
    return out


def test_close_leaves_no_workers_idle():
    d = make()
    pids = [d._whole_process.pid, d._live_process.pid]
    assert len(alive(*pids)) == 2
    d.close()
    assert alive(*pids) == []


def test_close_stops_a_busy_worker_and_the_call_in_it_fails():
    d = make()
    pid = d._whole_process.pid
    failed = []
    caller = threading.Thread(target=lambda: failed.append(_expect_error(lambda: d.whole(audio(2)))))
    caller.start()
    time.sleep(0.4)  # the recording is under way in the child
    d.close()
    caller.join(5)
    assert alive(pid) == [] and failed and "shut down" in failed[0]


def _expect_error(fn) -> str:
    try:
        fn()
    except RuntimeError as error:
        return str(error)
    return ""


def test_nothing_restarts_after_close():
    d = make()
    d.close()
    with pytest.raises(RuntimeError, match="shut down"):
        d.whole(audio(0))
    assert multiprocessing.active_children() == []


class FailsFor:
    """A factory that cannot build the models for one role."""

    def __init__(self, role_that_fails, message="model file corrupt"):
        self.role, self.message = role_that_fails, message

    def __call__(self, role):
        if role == self.role:
            raise ValueError(self.message)
        return Models(role)


def test_a_half_built_diarizer_leaves_no_child_running():
    with pytest.raises(RuntimeError):
        Diarizer(Path("unused"), factory=FailsFor("calls"))  # the recordings child had started
    assert multiprocessing.active_children() == []


def test_the_real_reason_a_child_could_not_start_is_reported():
    with pytest.raises(RuntimeError, match="model file corrupt"):
        Diarizer(Path("unused"), factory=FailsFor("recordings"))
    assert multiprocessing.active_children() == []


def test_a_cancelled_call_stops_the_work_and_the_next_call_works():
    d = make()
    try:
        cancel = threading.Event()
        pid = d._whole_process.pid
        threading.Timer(0.3, cancel.set).start()
        started = time.perf_counter()
        with pytest.raises(Cancelled):
            d.whole(audio(2), cancel=cancel)  # would take HOLD_S
        assert time.perf_counter() - started < HOLD_S - 0.3
        assert alive(pid) == []
        assert d.whole(audio(0))  # a new process took over
    finally:
        d.close()
