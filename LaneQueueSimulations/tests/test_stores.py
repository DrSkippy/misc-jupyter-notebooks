#!/usr/bin/env python
# coding: utf-8
import unittest
from lane_queues.stores import *


class DummyShopper:

        def __init__(self):
            self.join_time = 0
            self.checkout_time = 0


class LineTestCase(unittest.TestCase):

    def test_line(self):
        s = Line("mylabel")
        self.assertEqual(s.label, "mylabel")
    
    def test_len(self):
        s = Line("one")
        s.join_line(1, DummyShopper())
        
    def test_join_line(self):
        s = Line("one")
        s.join_line(1, DummyShopper())
        self.assertEqual(len(s), 1)
        s.join_line(1, DummyShopper())
        self.assertEqual(len(s), 2)

    def test_checkout(self):
        s = Line("one")
        s.join_line(1, DummyShopper())
        self.assertEqual(len(s), 1)
        a = s.checkout(2)
        self.assertEqual(len(s), 0)
        self.assertEqual(a.join_time, 1)
        self.assertEqual(a.checkout_time, 2)
        a = s.checkout(3)
        self.assertIsNone(a)
        
    def test_str(self):
        t = Line("two")
        t.join_line(1, DummyShopper())
        self.assertEqual(len(t), 1)
        b = str(t)
        self.assertEqual(b, "line: two\n  shoppers:  1*\n  ")


class ShopperTestCase(unittest.TestCase):

    def test_wait(self):
        s = Shopper()
        s.join_time = 1
        s.checkout_time = 4
        self.assertEqual(s.wait(), 3)
    
    def test_decision(self):
        sfl = [Line("a").join_line(1, Shopper()).join_line(2, Shopper()),
               Line("b").join_line(2, Shopper()).join_line(3, Shopper()).join_line(3, Shopper()),
               Line("c").join_line(4, Shopper())]
        s = Shopper()
        self.assertEqual(s.decision(1, sfl).label, "c" )

        sfl1 = [Line("a").join_line(1, Shopper()),
                Line("b").join_line(2, Shopper()).join_line(3, Shopper()).join_line(3, Shopper()),
                Line("c").join_line(4, Shopper())]
        s = Shopper()
        self.assertEqual(s.decision(1, sfl1).label, "a" )


class StoreTestCase(unittest.TestCase):

    def test_random_line(self):
        s = Store(3)
        self.assertEqual(len(s.lines), 3)
        a = [x.label for i in range(1000) for x in s.random_line()]
        a0 = a.count("line_0")
        a1 = a.count("line_1")
        self.assertGreater(a0/a1, 0.8)
        self.assertLess(a0/a1, 1.2)

    def test_index_line(self):
        s = Store(7)
        self.assertEqual(len(s.lines), 7)
        self.assertEqual(s.index_line([0,])[0].label, "line_0")

    def test_checkout(self):
        s = Store(3)
        s.lines[1].join_line(1, Shopper())
        s.lines[2].join_line(1, Shopper())
        s.lines[2].join_line(2, Shopper())
        s.lines[2].join_line(3, Shopper())
        a = s.checkout(5, [False, False, False])
        self.assertEqual(len(a), 3)
        self.assertTrue(all([x is None for x in a]))
        s.lines[0].join_line(1, Shopper())
        s.lines[1].join_line(1, Shopper())
        a = s.checkout(5, [True, True, True])
        self.assertEqual(len(a), 3)
        self.assertTrue(all([x is not None for x in a]))
        a = s.checkout(5, [False, True, True])
        self.assertEqual(len(a), 3)
        self.assertIsNone(a[0])
        self.assertIsNotNone(a[1])
        self.assertIsNotNone(a[2])

    def test_random_checkout(self):
        s = Store(3)
        s.lines[1].join_line(1, Shopper())
        s.lines[2].join_line(1, Shopper())
        s.lines[2].join_line(2, Shopper())
        s.lines[2].join_line(3, Shopper())
        a = s.random_checkout(5, p=1)
        self.assertEqual(len(a), 3)
        self.assertIsNone(a[0])
        s.lines[0].join_line(1, Shopper())
        s.lines[1].join_line(1, Shopper())
        a = s.random_checkout(5, p=0)
        self.assertEqual(len(a), 3)
        self.assertTrue(all([x is None for x in a]))
        a = s.random_checkout(5, p=0.5)
        self.assertEqual(len(a), 3)

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

