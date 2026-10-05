import multiprocessing as mp

def run_task(q):
    import ctypes
    # simulate segfault
    ctypes.string_at(0)

def safe_run():
    q = mp.Queue()
    p = mp.Process(target=run_task, args=(q,))
    p.start()
    p.join()
    print("Exit code:", p.exitcode)

if __name__ == '__main__':
    safe_run()
