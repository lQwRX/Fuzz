# fuzzer/logger.py
import time

class FuzzLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.f = open(log_file, 'w')
        self.f.write("timestamp\tindex\thex_msg\tstatus\tresponse\n")

    def log(self, idx, hex_msg, status, response):
        self.f.write(f"{time.time()}\t{idx}\t{hex_msg}\t{status}\t{response}\n")
        self.f.flush()

    def close(self):
        self.f.close()