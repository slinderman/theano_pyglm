import sys
import time
from IPython.display import clear_output

def wait_watching_stdout(ar, interval=1, truncate=100):
    """ Print the outputs of each worker 
    """
    stdoffs = [0]*len(ar.stdout)
    while not ar.ready():
        stdouts = ar.stdout
        if not any(stdouts):
            continue

        for eid, stdout in enumerate(stdouts):
            if stdout:
                newoff = len(stdout)
                if newoff > stdoffs[eid]:
                    print "[ stdout %2i ]\n%s" % (eid, stdout[stdoffs[eid]:])
                    stdoffs[eid] = newoff

        print '-' * 30
        print "%.3fs elapsed. Progress: %d/%d" % (ar.elapsed, ar.progress, len(ar))
        print ""

        sys.stdout.flush()
        time.sleep(interval)
