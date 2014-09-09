import os

def _find_profile(profile_name):
    from IPython.utils.path import locate_profile, get_ipython_dir
    try:
        profile_path = locate_profile(profile_name)
        exists = True
    except IOError:
        profile_path = os.path.join(get_ipython_dir(), "profile_%s"%profile_name)
        exists = False
    return profile_path, exists
    
def unique_profile_name():
    import hashlib
    import time
    return "local_%s" % hashlib.sha1(str(time.time())).hexdigest()[:8]


def get_engines(n_workers=4, profile_name="default", create_if_necessary=True, max_wait=90, ping_interval=1):
    """
    Select or start an IPython parallel cluster in code.
    """
    
    import select
    import time
    import subprocess
    from IPython.parallel import Client

    profile_dir, exists = _find_profile(profile_name)
    if not exists and create_if_necessary:
        # If the profile doesn't exist, create it please.
        print "Creating profile"
        os.system("ipython profile create %s --parallel" % profile_name)

    try:
        dviews = Client(profile=profile_name)[:]
        if len(dviews) != n_workers:
            print "Number of workers is %d, different than requested %d" % (len(dviews), n_workers)
            print "Stopping the cluster, restarting with proper number of workers"
            os.system("ipcluster stop")
            raise IOError
        else:
            return dviews
    except (IOError):
        print "Starting %d engines" % n_workers
        command = ["ipcluster", "start", "-n", "%d"%n_workers, '--profile=%s'%profile_name]
        print " ".join(command)
        p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        poll_obj = select.poll()
        poll_obj.register(p.stdout, select.POLLIN)   
    
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            p.stdout.flush()
            poll_result = poll_obj.poll(0)
            if poll_result:
                line = p.stdout.readline()
                if "IPClusterStart" in line:
                    print line
                    if "Engines appear to have started successfully" in line:
                        time.sleep(ping_interval)
                        print line
                        print profile_name
                        dviews = Client(profile=profile_name, timeout=time.time() - start_time)[:]
                        return dviews
                else:
                    time.sleep(ping_interval)

    print "Timed out."
    return None

def stop_engines(profile_name, max_wait=10, ping_interval=1):
        import select
        import time
        import subprocess
        print "Stopping the engines"
        command = ["ipcluster", "stop", "--profile=%s"%profile_name]
        p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        poll_obj = select.poll()
        poll_obj.register(p.stdout, select.POLLIN)   
    
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            p.stdout.flush()
            poll_result = poll_obj.poll(0)
            if poll_result:
                line = p.stdout.readline()
                if "IPClusterStop" in line:
                    if "CRITICAL" in line:
                        break
                    elif "Removing pid file" in line:
                        time.sleep(1.0)
                        break
            else:
                time.sleep(ping_interval)
