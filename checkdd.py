import sys
import collections
from functools import reduce
import itertools
import math

import numpy
import scipy
from scipy.special import *

def pairs(list_pair, lengths):
    #print(f"pair_list={list_pair},lengths={lengths}")
    list1 = itertools.combinations(list_pair[0], lengths[0])
    list2 = itertools.combinations(list_pair[1], lengths[1])
    res = itertools.starmap(lambda f, s: f + s, itertools.product(list1, list2))
    #print(f"res={list(ret)}")
    return res

#
# Check whether the given design is a dropout design or not
#
def checkdd(design):
    #print(f"design={design}")
    length = len(design)
    assert 1 < length and 0 < len(design[0])
    #print(f"length={length}")
    dd_results = []
    for t in range(len(design[0])-1):
        # t,t+1番目のブロックを取り出す
        neigborhoods = [(blocks[t], blocks[t+1]) for blocks in design]
        # print(f"neigborhoods={neigborhoods}")
        # t,t+1列の要素数
        n1 = len(collections.Counter(list(itertools.chain.from_iterable([n[0] for n in neigborhoods]))).keys())
        n2 = len(collections.Counter(list(itertools.chain.from_iterable([n[1] for n in neigborhoods]))).keys())
        #print(f"n1={n1}, n2={n2}")
        for k, l in itertools.product(range(1, 3), range(1, 3)):
            #print(f"(t,k,l)=({t+1},{k},{l})")
            # t,t+1番目のブロックを取り出して、任意の(k,l)-tupleを組み合わせたリストresを作る
            res = reduce(lambda lst, n: lst + list(pairs(n, (k, l))), neigborhoods, [])
            #print(f"res={list(res)}")
            res2 = list(collections.Counter(list(res)).items())
            res2.sort(key = lambda e: e[0])
            #print(f"res2={res2}")
            # 部分集合の数が一致するかどうか、Tallyの数が等しいかを調べる
            num_subset = scipy.special.binom(n1, k) * scipy.special.binom(n2, l)
            #print(f"len(res2)={len(res2)}, num={num_subset}")
            if len(res2) == num_subset and (len(res2) <= 1 or all([res2[i][1] == res2[i+1][1] for i in range(len(res2)-1)])):
                # 確認しているレイヤー番号二つ、見ている要素数、会合数 \[Lambda]を 出力
                dd_results.append((t, t + 1, k, l, res2[0][1]))
                print(f"(t, t+1 ,{{k,l}})=({t+1}, {t+2} ,{{{k},{l}}}), lambda={res2[0][1]}")
    dd_results
