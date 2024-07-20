import argparse
import collections
import itertools
import math
import operator
import os
import random
import numpy as np
import sympy
import time

import FiniteField
from FiniteField import *

import multiprocessing as mlp
from multiprocessing import Pool

import xml.etree.ElementTree as ET

####
## Finite Projective Geometry Libraries

def affineGeometry(field):
    p = field.PrimitivePolynomial.field.p
    class AffineSpace:
        Field = field
        def __init__(self):
            self.Points = field.MemberCoefficients
        def t_flats(self, t):
            assert 0 < t < len(self.Points)
            t_flat_hyperplane = np.matrix(self.Points[1:1+t])
            #print(f't_flat_hyperplane=\n{t_flat_hyperplane}')
            t_flat_first = [(np.matrix(base) * t_flat_hyperplane).tolist()[0]
                                for base in itertools.product(range(p), repeat = t)]
            yield t_flat_first
            complements = list(filter(lambda e: not(e in t_flat_first), self.Points))
            # print(f'complements={complements}')
            while len(complements) > 0:
                addpoint = complements[0]
                # print(f'addpoint={addpoint}')
                t_flat = [ (np.matrix(v) + np.matrix(addpoint)).tolist()[0] for v in t_flat_first]
                complements = [e for e in complements if not(e in t_flat)]
                yield t_flat
        def t_flats_ids(self, t):
            for flat in self.t_flats(t):
                yield [ self.Points.index(v) + 1 for v in flat]
        def hyperplanes(self):
            trace = field.tracefun()
            #print(f'trace={trace}')
            index_values = list(enumerate(trace.evaluate((field.Alpha **i).poly) % field.PrimitivePolynomial for i in  range((field.P ** field.PrimitivePolynomial.degree()) - 1)))
            # print(f'index_values={index_values}')
            for i in range(field.P):
                hyperplane = [index for index, value in index_values if value == i]
                #print(f'hyperplanes[{i}]={list(hyperplane)}')
                yield hyperplane
    AffineSpace.__name__ = 'AffineSpace(' + str(field.P) + '^'  + str(field.Dimension) + ' [' + str(field.PrimitivePolynomial) + '])'
    return AffineSpace()

def projectiveSpace(field):
    class ProjectiveSpace:
        Field = field
        def __init__(self):
            self.Points = list(self.__makePoints())
            assert len(self.Points) == int(QBinomial(field.PrimitivePolynomial.degree(), 1, field.P))
        @classmethod
        def __makePoints(cls):
            points = []
            for av in (x for x in field.MemberCoefficients if x != field.Zero.poly):
                def scales():
                    for i in range(1, field.P):
                        yield [x * i for x in av]
                def intersects():
                    return any(x == y for x, y in itertools.product(scales(), points))
                if not intersects():
                    points.append(av)
                    yield av
    ProjectiveSpace.__name__ = 'ProjectiveSpace(' + str(field.P) + '^' + str(field.Dimension) + ' [' + str(field.PrimitivePolynomial) + '])'
    return ProjectiveSpace()

# Concurrent procedure
def __createBlocks(params):
    i, dots, t_flats = params
    print('Computing ' + str(i) + '-th blocks..')
    startTime = time.time()
    blocks = []
    for hyperplane in ([j + 1 for j, w in enumerate(v) if w == i] for v in dots):
        block = [[ x for x in hyperplane if x in t_flat] for t_flat in t_flats]
        sizes = (len(b) for b in block)
        if all(a != 0 and a == b for a, b in itertools.product(sizes, repeat=2)):
            blocks.append(block)
    print(str(i) + '-th blocks computed: ' + str(time.time() - startTime) + ' sec')
    return blocks

def makeDropoutDesign_1_2(primitivePolynomial, t, sequential = False):
    dim = primitivePolynomial.degree()
    p = primitivePolynomial.field.p

    print('Computing extension field..')
    field = extendedFiniteField(primitivePolynomial)()
    # print(f'field=\n{field.MemberCoefficients},\nlen={len(field.MemberCoefficients)}')

    print('Computing points on affine space..')
    affine = affineGeometry(field)
    # print(f'affine=\n{affine.Points},\nlen={len(affine.Points)}')

    print('Computing ' + str(t) + '-flats..')
    startTime = time.time()
    t_flats = list(affine.t_flats_ids(t))
    print(str(t) + '-flats constructed(' + str(len(t_flats)) + '): ' + str(time.time() - startTime) + ' sec')
    # print(f'{t}-flats=\n{t_flats},\n{len(t_flats)}')

    print('Computing points on projective space..')
    startTime = time.time()
    projective = projectiveSpace(field)
    print('Projective space constructed: ' + str(time.time() - startTime) + ' sec')
    # print(f'projective=\n{projective.Points},\nlen={len(projective.Points)}')

    print('Computing dots between affine and projective..')
    startTime = time.time()
    afpoints = [ np.array([int(x) for x in pt]) for pt in affine.Points]
    prpoints = (np.array([int(x) for x in pt]) for pt in projective.Points)
    dots = [[int(np.dot(i, j)) % p for j in afpoints] for i in prpoints]
    print('Dot computed(' + str(len(dots)) + '): ' + str(time.time() - startTime) + ' sec')

    print('Computing blocks..')
    if sequential:
        return itertools.chain.from_iterable(list(map(__createBlocks, ((i, dots, t_flats) for i in range(p)))))
    else:
        cpuCount = max(1, mlp.cpu_count() - 1)
        print('# of cpus used = ' + str(cpuCount))
        with Pool(cpuCount) as pool:
            startTime = time.time()
            result = pool.map(__createBlocks, ((i, dots, t_flats) for i in range(p)))
            print('Blocks computed: ' + str(time.time() - startTime) + ' sec')
            return itertools.chain.from_iterable(result)

def to_params_dropout_design_1_2(dim, p, t):
    # return (v, k, lambda, n, b)
    return (p**t, p**(t-1), (p**(dim-2) - p**(dim-t-1))//(p-1), p**(dim-t), (p*(p**dim - p**(dim-t)))//(p-1))

def to_setting_dropout_design_1_2(v, n):
    # print(f'to_setting_dropout_design_1_2(v={v}, n={n})')
    if v == None or v < 1 or n == None or n < 1:
        raise Exception('Invalid inputs(v=' + str(v) + ',n=' + str(n) + ')')
    fac = sympy.factorint(v)
    if len(fac) != 1:
        raise Exception('Is not power of prime(v=' + str(v) + 'is ' + str(fac))
    p = list(fac.keys())[0]
    t = fac[p]
    dim = math.log(n * v, p)
    if not math.isclose(dim, np.rint(dim)):
        raise Exception('Dimension is not integer(dim=' + str(dim) + ')')
    if int(np.rint(dim)) <= t:
        raise Exception('Invalid Dimensions(dim=' + str(int(np.rint(dim))) + ') <= t=' + str(t) + ')')
    return (int(np.rint(dim)), p, t)

def layers_to_dropout_design_1_2(n):
    for v in range(1, min(100000, n**10)):
        try:
            dim, p, t = to_setting_dropout_design_1_2(v, n)
            _, k, lamb, _, b = to_params_dropout_design_1_2(dim, p, t)
            yield v, k, lamb, n, b, dim, p, t
        except:
            pass

def units_to_dropout_design_1_2(v):
    for n in range(1, min(100000, v**10)):
        try:
            dim, p, t = to_setting_dropout_design_1_2(v, n)
            _, k, lamb, _, b = to_params_dropout_design_1_2(dim, p, t)
            yield v, k, lamb, n, b, dim, p, t
        except:
            pass

def allsettings_dropout_design_1_2(upper):
    for n, v in itertools.product(range(1, upper+1), repeat=2):
        try:
            dim, p, t = to_setting_dropout_design_1_2(v, n)
            _, k, lamb, _, b = to_params_dropout_design_1_2(dim, p, t)
            yield v, k, lamb, n, b, dim, p, t
        except:
            pass

# dropout design
def dropout_designs_1_2(dim, p, t, sequential = False):
    print('dropout_designs_1_2(dim=' + str(dim) + ', p=' + str(p) + ', t=' + str(t) + ')')
    v, k, lamb, n, b = to_params_dropout_design_1_2(dim, p, t)
    print('v=' + str(v) + ', k=' + str(k) + ', lambda=' + str(lamb) + ', n=' + str(n) + ', b=' + str(b))
    print('Computing primitive polynomial..')
    irrs = irreduciblePolynomials(modulus=p, degree=dim)()
    primitives = primitivePolynomials(irrs)()
    del irrs
    class DropoutDesign:
        __path = './designs/dd_' + str(dim) + '_' + str(p) + '_' + str(t) + '_*.xml'
        Dim = dim
        P = p
        T = t
        VarietiesCount = v
        K = k
        Lamb = lamb
        N = n
        BlockCount = b
        Count = primitives.Count
        def __init__(self):
            filePath = os.path.dirname(self.__path)
            if not os.path.exists(filePath):
                os.makedirs(filePath)
            self.__designs = []
        def get(self, i):
            if i < 0 or self.Count <= i:
                return None
            if i < len(self.__designs):
                return self.__designs[i]
            def write(dropoutdesign, file):
                print('Writing design to ' + str(file) + '..')
                design = ET.Element("design")
                for nb, blc in enumerate(dropoutdesign):
                    block = ET.Element("block", attrib={'nb': str(nb)})
                    for nsub, subblc in enumerate(blc):
                        subblock = ET.Element("subblock", attrib={'nsub': str(nsub)})
                        for npt, pnt in enumerate(subblc):
                            point = ET.Element("point", attrib={'npt':str(npt)})
                            point.text = str(pnt)
                            subblock.append(point)
                        block.append(subblock)
                    design.append(block)
                root = ET.ElementTree(design)
                root.write(file, encoding='utf-8')
            def read(i):
                file = self.__path.replace('*', str(i))
                print('Reading design from ' + str(file) + '..')
                try:
                    tree = ET.parse(file)
                    design = tree.getroot()
                    assert design.tag == 'design'
                    dsgn = []
                    for block in design.findall(".//block"):
                        blc = []
                        for subblock in block.findall(".//subblock"):
                            blc.append([int(pt.text) for pt in sorted(subblock.findall("./point"), key=lambda pt: int(pt.attrib['npt']))])
                        dsgn.append(blc)
                    return dsgn
                except:
                    print('An error occurs when opening ' + str(file) + '.')
                    dsgn = list(makeDropoutDesign_1_2(primitives.get(i), t,sequential))
                    write(dsgn, file)
                    return dsgn
            for j in range(len(self.__designs), i + 1):
                self.__designs.append(read(j))
            return self.__designs[len(self.__designs) - 1]
        def getAll(self):
            return (self.get(i) for i in range(self.Count))
    DropoutDesign.__name__ = '(' + str(dim) + ', ' + str(p) + ', ' + str(t) + ')-DropoutDesign'
    return DropoutDesign()

def dropout_design_1_2(dim, p, t, sequential = False):
    print('dropout_design_1_2(dim=' + str(dim) + ', p=' + str(p) + ', t={t})')
    designs = dropout_designs_1_2(dim, p, t, sequential)
    return ((designs.VarietiesCount, designs.K, designs.Lamb, designs.N, designs.BlockCount), designs.get(0))

def dropout_design_1_2_to_dropout(design):
    layers = [[[] for _ in block] for block in design]
    for j in range(0, len(design[0])): # #layers
        layernums = set(itertools.chain.from_iterable({e for e in block[j]} for block in design))
        rule = { j: i for i, j in enumerate(layernums)}
        for i, layer in enumerate(sorted(rule[e] for e in block[j]) for block in design):
            layers[i][j] = list(layer)
    return layers

def shift_dropout_varieties(dropout, shift = 1):
    if shift == None or shift == 0:
        return dropout
    modulo = max(itertools.chain.from_iterable(itertools.chain.from_iterable(dropout))) + 1
    assert 0 < modulo
    if shift % modulo == 0:
        return dropout
    return [[list(sorted((elem + shift) % modulo for elem in batch)) for batch in batches] for batches in dropout]

def shift_dropout_blocks(dropout, shift = 1):
    if shift == None or shift == 0:
        return dropout
    return dropout[len(dropout) - (shift % len(dropout)):] + dropout[:len(dropout) - (shift % len(dropout))]

def rand_dropout_varieties(dropout):
    length = max(itertools.chain.from_iterable(itertools.chain.from_iterable(dropout))) + 1
    assert 0 < length
    perm = random.sample(range(length), length)
    return [[list(sorted(perm[var] for var in batch)) for batch in batches] for batches in dropout]

def rand_dropout_blocks(dropout):
    return random.sample(dropout, len(dropout))

#############################################
# Run
if __name__ == "__main__":
    import checkdd
    from checkdd import *
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dimension", type = int, default = 5, help="dimension of primitive polynomial")
    parser.add_argument("-p", "--prime", type = int, default = 3, help="prime number")
    parser.add_argument("-t", "--tflat", type = int, default = 3, help="dimension of t-flats")
    parser.add_argument("-v", "--nunits", type = int, default = None, help="#units per layer (should be the power of prime!)")
    parser.add_argument("-n", "--nlayers", type = int, default = None, help="#layers")
    parser.add_argument("-a", "--all", type = int, default = None, help="available parameter sets")
    parser.add_argument("-o", "--out", type = str, default = 'dropoutdesign.txt', help="define the output path")
    args = parser.parse_args()

    def create(dimension, prime, tflat):
        startTime = time.time()
        design = list(dropout_design_1_2(dimension, prime, tflat, sequential=False)[1])
        dropout = dropout_design_1_2_to_dropout(design)
        elapsedTime = time.time() - startTime
        print('(' + str(dimension) + ', {' + str(prime) + ', {' + str(tflat) + ')-design constructed: ' + str(elapsedTime) + ' sec')
        # print('dropout=\n' + str(dropout)')
        with open(args.out, mode = 'w') as f:
            f.write('(' + str(dimension) + ', {' + str(prime) + ', {' + str(tflat) + ')-design\n')
            f.write(str(dropout).replace('[', '{').replace(']', '}').replace(', ', ','))
        print('Check design..')
        checkdd(dropout)
    create(args.dimension, args.prime, args.tflat)