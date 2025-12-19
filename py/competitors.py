'''
Blackbox Test Case Prioritization Baselines

This module implements blackbox TCP baselines that require no coverage information.
Originally adapted from ICSE'18 FAST implementation.

Implemented Methods:
- STR: String distance baseline (Ledru et al.) - ACTIVE
- I-TSD: Information-theoretic test suite diversity - NOT USED (inefficient)

Note: All whitebox methods have been removed. This codebase focuses on blackbox TCP.
I-TSD is kept for reference but not used in experiments due to poor efficiency.
'''

# BLACK BOX PRIORITIZATIONS
# STR (Ledru) - active
# I-TSD - reference only, not used in experiments

from collections import defaultdict
from collections import OrderedDict
from pickle import dump, load
import bz2
import itertools
import os
import random
import time
import scipy.special
import sys


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# BLACK BOX
def str_(input_file):
    def loadTestSuite(input_file):
        TS = {}
        maxlen = 0
        with open(input_file) as fin:
            tcID = 1
            for tc in fin:
                if len(tc) > maxlen:
                    maxlen = len(tc)
                TS[tcID] = tc[:-1]
                tcID += 1
        for tcID, tc in TS.items():
            asciivec = []
            for c in tc:
                asciivec.append(float(ord(c)))
            asciivec += [0.0] * (maxlen - len(tc))
            TS[tcID] = asciivec
        return OrderedDict(TS.items())

    def manhattanDistance(TCS, i, j):
        u, v = TCS[i], TCS[j]
        return sum([abs(float(ui) - float(vi)) for ui, vi in zip(u, v)])

    def storePairwiseDistance(TCS, sigfile):
        D = defaultdict(float)
        combs = scipy.special.binom(len(TCS.keys()), 2)
        iteration, total = 0, float(combs)
        for pair in itertools.combinations(TCS.keys(), 2):
            # print iteration, total
            iteration += 1
            if iteration % 100 == 0:
                sys.stdout.write("  Progress: {}%\r".format(
                    round(100*iteration/total, 2)))
                sys.stdout.flush()

            i, j = pair
            if i < j:
                D[(i, j)] = manhattanDistance(TCS, i, j)
        dump(D, open(sigfile, "wb"))

    def loadPairwiseDistance(sigfile):
        return load(open(sigfile, "rb"))

    def removeDuplicates(TCS):
        unique = set()
        P = []
        for tcID, tc in TCS.items():
            tc = tuple(tc)
            if tc in unique:
                P.append(tcID)
            else:
                unique.add(tc)
        for tc in P:
            del TCS[tc]
        return P

    def select(TCS, D, T):
        s, dist_s = 0, -1
        for ui in TCS:
            dist = float("Inf")
            for vi in T:
                if ui < vi and D[(ui, vi)] < dist:
                    dist = D[(ui, vi)]
            if dist > dist_s:
                s, dist_s = ui, dist
        return s

    # # # # # # # # # # # # # # # # # # # # # # #

    TCS = loadTestSuite(input_file)
    P1 = []

    sigfile = input_file.replace(".txt", "___.pickle").replace("___", "_distmatrix")
    if not os.path.exists(sigfile):
        ledru_t = time.perf_counter()
        storePairwiseDistance(TCS, sigfile)
        ledru_time = time.perf_counter() - ledru_t
        with open("{}_sigtime.txt".format(input_file.split(".")[0]), "w") as fout:
            fout.write(repr(ledru_time))
    else:
        with open("{}_sigtime.txt".format(input_file.split(".")[0]), "r") as fin:
            ledru_time = eval(fin.read().replace("\n", ""))

    ptime_start = time.perf_counter()
    load_time_start = time.perf_counter()
    D = loadPairwiseDistance(sigfile)

    P2 = removeDuplicates(TCS)

    s = select(TCS, D, TCS.keys())
    P1.append(s)
    del TCS[s]

    iteration, total = 0, float(len(TCS))
    while len(TCS) > 0:
        iteration += 1
        if iteration % 100 == 0:
            sys.stdout.write("  Progress: {}%\r".format(
                round(100*iteration/total, 2)))
            sys.stdout.flush()

        s = select(TCS, D, P1)
        P1.append(s)
        del TCS[s]

    ptime = time.perf_counter() - ptime_start

    return ledru_time, ptime, P1 + P2


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# I-TSD
def i_tsd(input_file):
    def loadTestSuite(input_file):
        TS = {}
        with open(input_file) as fin:
            tcID = 1
            for tc in fin:
                TS[tcID] = tc[:-1]
                tcID += 1
        shuffled = list(TS.items())
        random.shuffle(shuffled)
        TS = OrderedDict(shuffled)
        return TS

    def compressExcept(TCS, toExclude):
        s = " ".join([TCS[tcID] for tcID in TCS.keys() if tcID != toExclude])
        cs = bz2.compress(s.encode('utf-8'))
        return sys.getsizeof(cs)

    def select(TCS):
        maxIndex, maxCompress = 0, 0
        for tcID in TCS.keys():
            c = compressExcept(TCS, tcID)
            if c > maxCompress:
                maxIndex, maxCompress = tcID, c
        return maxIndex

    # # # # # # # # # # # # # # # # # # # # # # # # # # #

    stime = 0.0
    ptime_start = time.perf_counter()
    TCS = loadTestSuite(input_file)

    P = [0]

    iteration, total = 0, float(len(TCS))
    while len(TCS) > 0:
        iteration += 1
        if iteration % 100 == 0:
            sys.stdout.write("  Progress: {}%\r".format(
                round(100*iteration/total, 2)))
            sys.stdout.flush()

        s = select(TCS)
        P.append(s)
        del TCS[s]

    ptime = time.perf_counter() - ptime_start

    return stime, ptime, P[1:]


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
