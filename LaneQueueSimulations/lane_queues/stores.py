#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import random


class Line():

    def __init__(self, label):
        self.label = label
        self.shoppers = []
        
    def __len__(self):
        return len(self.shoppers)
    
    def join_line(self, t, shopper):
        shopper.join_time = t
        self.shoppers.append(shopper)
        return self
    
    def checkout(self, t):
        shopper_checked_out = None
        if len(self.shoppers) > 0:
            shopper_checked_out = self.shoppers.pop(0)
            shopper_checked_out.checkout_time = t
        return shopper_checked_out
        
    def __str__(self):
        res = "\n  ".join([
            "line: {}".format(self.label),
            "shoppers: {:2}{}".format(len(self.shoppers), "*"*len(self.shoppers)),
            ""
        ])
        return res


class Shopper():
    
    def __init__(self):
        self.join_time = 0
        self.checkout_time = 0
    
    def decision(self, t, select_from_lines):
        idx = -1
        min_size = 1000000
        for i, l in enumerate(select_from_lines):
            if len(l) < min_size:
                idx = i
                min_size = len(l)
        return select_from_lines[idx]
        
    def wait(self):
        return self.checkout_time - self.join_time


class Store():
        
    def __init__(self, n=5):
        self.lines = [Line("line_{}".format(i)) for i in range(n)]
        self.metrics = []
        self.customer_wait_metrics = []
        
    def random_line(self, k=1):
        return random.sample(self.lines, k)

    def index_line(self, indexes):
        return [self.lines[i] for i in indexes]

    def checkout(self, t, line_checkout_status):
        checked_out_shoppers = []
        for i, l in enumerate(self.lines):
            if line_checkout_status[i]:
                checked_out_shoppers.append(l.checkout(t))
            else:
                checked_out_shoppers.append(None)
        return checked_out_shoppers

    def random_checkout(self, t, p):
        line_checkout_status = np.random.random(len(self.lines)) < p
        return self.checkout(t, line_checkout_status)

    def report(self, t, cos, output=False):
        self.customer_wait_metrics = [c.wait() for c in cos if c is not None]
        avg_n = np.average([len(l) for l in self.lines])
        avg_t = [np.average([t - shopper.join_time for shopper in l.shoppers]) 
                 for l in self.lines]
        avg_wait = np.average(self.customer_wait_metrics)
        self.metrics.append([t, avg_n, avg_wait]+avg_t)
        for line in self.lines:
            print(line) if output else None
