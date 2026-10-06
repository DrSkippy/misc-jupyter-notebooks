""" Load submitted strategies without copying them over strategy.py.

    Submissions are saved next to this file as "strategy.py.<Name>",
    e.g. "strategy.py.Jeff". Refer to one by its name ("Jeff") or by
    a path to any file that defines bet_strategy.
"""
import glob
import importlib.machinery
import importlib.util
import os
import sys

SUBMISSION_PREFIX = "strategy.py."
HERE = os.path.dirname(os.path.abspath(__file__))


def list_submissions(folder=HERE):
    """Return {name: path} for every strategy.py.<Name> file in folder."""
    paths = sorted(glob.glob(os.path.join(folder, SUBMISSION_PREFIX + "*")))
    return {os.path.basename(p)[len(SUBMISSION_PREFIX):]: p for p in paths}


def resolve(name_or_path, folder=HERE):
    """Return the file for a submission name or a path."""
    if os.path.isfile(name_or_path):
        return name_or_path
    path = os.path.join(folder, SUBMISSION_PREFIX + name_or_path)
    if os.path.isfile(path):
        return path
    raise FileNotFoundError("No submission '{}'. Available: {}".format(
        name_or_path, ", ".join(list_submissions(folder))))


def load_strategy(name_or_path, folder=HERE):
    """Import a submission file and return its bet_strategy function."""
    path = resolve(name_or_path, folder)
    # submissions do "from sim_parameters import *"
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    module_name = "submission_" + os.path.basename(path).replace(".", "_")
    loader = importlib.machinery.SourceFileLoader(module_name, path)
    spec = importlib.util.spec_from_loader(module_name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module.bet_strategy


def load_all(folder=HERE):
    """Return {name: bet_strategy} for every submission in folder."""
    return {name: load_strategy(path) for name, path in list_submissions(folder).items()}
