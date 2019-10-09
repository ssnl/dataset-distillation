import atexit
import logging
import multiprocessing as mp
import sys
import traceback


class ExceptionWrapper(object):
    r"""Wraps an exception plus traceback to communicate across threads"""

    def __init__(self, exc_info):
        self.exc_type = exc_info[0]
        self.exc_msg = "".join(traceback.format_exception(*exc_info))

    def reconstruct(self):
        return self.exc_type(self.exc_msg)


class Process(mp.Process):
    def __init__(self, *args, **kwargs):
        mp.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None

    def run(self):
        try:
            mp.Process.run(self)
            self._cconn.send(None)
        except Exception:
            self._cconn.send(ExceptionWrapper(sys.exc_info()))
            raise

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


class FixSizeProcessQueue(object):
    def __init__(self, size):
        self.size = size
        self.queue = [None for _ in range(size)]
        self.at = 0
        atexit.register(self.join_all)

    def join(self, idx):
        if self.queue[idx] is not None:
            p = self.queue[idx]
            p.join()
            if p.exception:
                logging.error('Previous subprocess ended with error.')
                raise p.exception.reconstruct()
            self.queue[idx] = None

    def enqueue(self, fn, *args, **kwargs):
        p = Process(target=fn, args=args, kwargs=kwargs)
        p.daemon = True
        self.join(self.at)
        p.start()
        self.queue[self.at] = p
        self.at = (self.at + 1) % self.size

    def join_all(self):
        for i in range(self.size):
            self.join(i)

    def __del__(self):
        self.join_all()
