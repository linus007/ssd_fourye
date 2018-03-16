import Queue
import threading
class ConcurrentQueue(object):
    def __init__(self, capacity=0):
        self.__capacity = capacity

        self.__mutext = threading.Lock()

        self.__cond = threading.Condition(self.__mutext)

        self.__queue = Queue.Queue(maxsize=self.__capacity)

    def put(self, elem):
        if self.__cond.acquire():

            if self.__queue.qsize() < self.__capacity:
                self.__queue.put(elem)

            self.__cond.notify()
            self.__cond.release()

    def get(self):
        if self.__cond.acquire():
            if self.__queue.empty():
                elem = None
            else:
                elem = self.__queue.get()
            self.__cond.notify()
            self.__cond.release()
        return elem

    def clear(self):
        if self.__cond.acquire():
            self.__queue.queue.clear()
            self.__cond.release()
            self.__cond.notify_all()

    def size(self):
        size = 0
        if self.__mutext.acquire():
            size = self.__queue.qsize()
            self.__mutext.release()

        return size
def debug():
    q = ConcurrentQueue(10);
    q.put(10);
    q.put(10);
    q.put(10);
    q.put(1);
    q.put(1);
    q.put(1);
    q.put(1);
    q.put(0);
    q.put(0);
    q.put(0);
    q.put(0);
    q.put(0);
    print(q.size())
    print(q.get())
    print(q.size())

if __name__ == '__main__':
    if 1:
        debug()