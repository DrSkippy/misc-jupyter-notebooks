import pandas as pd

from lane_queues.stores import *

def random_worker(args):
    d, lines, total_time, p_shopper, p_checkout = args
    print("lines: {}\nlines to inspect: {}\n\n".format(lines, d))
    s = Store(lines)
    for t in range(total_time):
        if random.random() < p_shopper:
            sh = Shopper()
            lines_select = s.random_line(d)
            sh.decision(t, lines_select).join_line(t, sh)
        cos = s.random_checkout(t, p_checkout)
        s.report(t, cos)
    df = pd.DataFrame.from_records(s.metrics)
    df.columns=["time",
                "avg_shoppers",
                "avg_wait"] + [
                "line_{}".format(x) for x in range(lines)]
    return df, d

def simulation_lines(n_steps, n_lines):
    # pre calculate the order of all of the lines to be checked
    lines = list(range(n_lines))
    lines_ar = np.zeros((n_steps, n_lines), dtype=int)
    for i in range(n_steps):
        np.random.shuffle(lines)
        lines_ar[i] = lines
    return lines_ar

def simulation_prob_arrive(n_steps, p_arrive):
    # precalculate arrival state
    res = np.ones(n_steps, dtype=int)*(np.random.random_sample(n_steps) < p_arrive)
    return res

def simulation_prob_checkout(n_steps, n_lines, p_checkout):
    # precalculate arrival state
    res = np.ones((n_steps, n_lines), dtype=int)*(np.random.random_sample((n_steps, n_lines)) < p_checkout)
    return res

def lockstep_worker(args):
    d, lines, total_time, sl, spa, spc = args
    print("lines: {}\nlines to inspect: {}\n\n".format(lines, d))
    s = Store(lines)
    for t in range(total_time):
        if spa[t]:
            sh = Shopper()
            lines_select = s.index_line(sl[t][:d])
            sh.decision(t, lines_select).join_line(t, sh)
        cos = s.checkout(t, spc[t])
        s.report(t, cos)
    df = pd.DataFrame.from_records(s.metrics)
    df.columns=["time",
                "avg_shoppers",
                "avg_wait"] + [
                "line_{}".format(x) for x in range(lines)]
    return df, d