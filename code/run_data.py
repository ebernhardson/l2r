import subprocess
import time

class PythonRunner(object):
    def __init__(self, python="../bin/python", parallel=False, n_procs=4):
        self.python = python
        self.parallel = parallel
        self.n_procs = n_procs

    def __call__(self, cmds):
        if self.parallel:
            active_procs = []
            for cmd in cmds:
                full_cmd = [self.python] + cmd
                print('starting: %s' % (" ".join(full_cmd)))
                new_proc = subprocess.Popen(full_cmd)
                active_procs.append(new_proc)
                if len(active_procs) >= self.n_procs:
                    active_procs = self.wait_complete(active_procs)
            while len(active_procs) > 0:
                active_procs = self.wait_complete(active_procs)
        else:
            for cmd in cmds:
                full_cmd = [self.python] + cmd
                print('starting: %s' % (" ".join(full_cmd)))
                subprocess.call(full_cmd)

    def wait_complete(self, procs):
        while True:
            for i, proc in enumerate(procs):
                status = proc.poll()
                if status == None:
                    continue
                elif status == 0:
                    # complete
                    del procs[i]
                    return procs
                else:
                    print "command failed with status", status
                    del procs[i]
                    return procs
            time.sleep(10)


if __name__ == "__main__":
    # --------------------------------------------------
    # Collect the data from hive
    PythonRunner()([
        ["data_prepare.py"]
    ])

    # --------------------------------------------------
    # Get the data to augment it with
    PythonRunner(parallel=True)([
        ["data_augment_es_docs.py"],
        ["data_augment_relevance.py"],
        ["data_augment_es_docs_termvec.py"],
        ["data_augment_es_query_termvec.py"],
    ])

    # --------------------------------------------------
    # Merge the data and augments into ALL_DATA
    PythonRunner()([
        ["data_merge_augment.py"],
    ])

    # --------------------------------------------------
    # Generate features
    PythonRunner(parallel=True)([
        # generate basic features
        ["feature_basic.py"],
        # generate distance features
        ["feature_distance.py"],
        # features from es docs and termvec apis
        ["feature_ident.py"],
        # cosine sim from es termvec api
        ["feature_vector_space.py"],
    ])

