from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, help="port to publish from the container")

# from python 3.3 source
# https://github.com/python/cpython/blob/master/Lib/shutil.py
def which(cmd, mode=os.F_OK | os.X_OK, path=None):
    """Given a command, mode, and a PATH string, return the path which
    conforms to the given mode on the PATH, or None if there is no such
    file.
    `mode` defaults to os.F_OK | os.X_OK. `path` defaults to the result
    of os.environ.get("PATH"), or can be overridden with a custom search
    path.
    """
    # Check that a given file can be accessed with the correct mode.
    # Additionally check that `file` is not a directory, as on Windows
    # directories pass the os.access check.
    def _access_check(fn, mode):
        return (os.path.exists(fn) and os.access(fn, mode)
                and not os.path.isdir(fn))

    # If we're given a path with a directory part, look it up directly rather
    # than referring to PATH directories. This includes checking relative to the
    # current directory, e.g. ./script
    if os.path.dirname(cmd):
        if _access_check(cmd, mode):
            return cmd
        return None

    if path is None:
        path = os.environ.get("PATH", os.defpath)
    if not path:
        return None
    path = path.split(os.pathsep)

    if sys.platform == "win32":
        # The current directory takes precedence on Windows.
        if not os.curdir in path:
            path.insert(0, os.curdir)

        # PATHEXT is necessary to check on Windows.
        pathext = os.environ.get("PATHEXT", "").split(os.pathsep)
        # See if the given file matches any of the expected path extensions.
        # This will allow us to short circuit when given "python.exe".
        # If it does match, only test that one, otherwise we have to try
        # others.
        if any(cmd.lower().endswith(ext.lower()) for ext in pathext):
            files = [cmd]
        else:
            files = [cmd + ext for ext in pathext]
    else:
        # On other platforms you don't have things like PATHEXT to tell you
        # what file suffixes are executable, so just pass on cmd as-is.
        files = [cmd]

    seen = set()
    for dir in path:
        normdir = os.path.normcase(dir)
        if not normdir in seen:
            seen.add(normdir)
            for thefile in files:
                name = os.path.join(dir, thefile)
                if _access_check(name, mode):
                    return name
    return None


def main():
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if not args[i].startswith("--"):
            break
        i += 2
    
    a = parser.parse_args(args[:i])
    cmd = args[i:]

    # check if nvidia-docker or docker are on path
    docker_path = which("nvidia-docker")
    if docker_path is None:
        docker_path = which("docker")
    
    if docker_path is None:
        raise Exception("docker not found")

    docker_args = [
        "--rm",
        "--volume",
        "/:/host",
        "--workdir",
        "/host" + os.getcwd(),
        "--env",
        "PYTHONUNBUFFERED=x",
        "--env",
        "CUDA_CACHE_PATH=/host/tmp/cuda-cache",
        "--env",
        "GOOGLE_PROJECT=" + os.environ.get("GOOGLE_PROJECT", ""),
        "--env",
        "GOOGLE_CREDENTIALS=" + os.environ.get("GOOGLE_CREDENTIALS", ""), 
    ]

    if a.port is not None:
        docker_args += ["--publish", "%d:%d" % (a.port, a.port)]

    args = [docker_path, "run"] + docker_args + ["affinelayer/pix2pix-tensorflow"] + cmd

    if not os.access("/var/run/docker.sock", os.R_OK):
        args = ["sudo"] + args

    os.execvp(args[0], args)


main()
