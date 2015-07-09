class Timer():
    def __init__(self, txt):
        self.txt = txt
 
    def __enter__(self):
        self.start = time()
        print(self.txt + '... ', end='')
        sys.stdout.flush()
 
    def __exit__(self, type, value, tb):
        print("{:.2f} sec.".format(time()-self.start))