##
# Finite Field module
# Changed modules below:
#  https://jeremykun.com/2014/03/13/programming-with-finite-fields/
#  https://github.com/j2kun/finite-fields

import collections
import fractions
from functools import reduce
import itertools
try:
    from itertools import zip_longest
except ImportError:
    from itertools import izip_longest as zip_longest
import operator
import random
import sympy
import xml.etree.ElementTree as ET

###
## Number Types

# memoize calls to the class constructors for fields
# this helps typechecking by never creating two separate
# instances of a number class.
def memoize(f):
   cache = {}
   def memoizedFunction(*args, **kwargs):
      argTuple = args + tuple(kwargs)
      if argTuple not in cache:
         cache[argTuple] = f(*args, **kwargs)
      return cache[argTuple]
   memoizedFunction.cache = cache
   return memoizedFunction

# type check a binary operation, and silently typecast 0 or 1
def typecheck(f):
   def newF(self, other):
      if (hasattr(other.__class__, 'operatorPrecedence') and
            other.__class__.operatorPrecedence > self.__class__.operatorPrecedence):
         return NotImplemented
      if type(self) is not type(other):
         try:
            other = self.__class__(other)
         except TypeError:
            message = 'Not able to typecast %s of type %s to type %s in function %s'
            raise TypeError(message % (other, type(other).__name__, type(self).__name__, f.__name__))
         except Exception as e:
            message = 'Type error on arguments %r, %r for functon %s. Reason:%s'
            raise TypeError(message % (self, other, f.__name__, type(other).__name__, type(self).__name__, e))
      return f(self, other)
   return newF

# require a subclass to implement +-* neg and to perform typechecks on all of
# the binary operations finally, the __init__ must operate when given a single
# argument, provided that argument is the int zero or one
class DomainElement(object):
   operatorPrecedence = 1
   # the 'r'-operators are only used when typecasting ints
   def __radd__(self, other): return self + other
   def __rsub__(self, other): return -self + other
   def __rmul__(self, other): return self * other
   # square-and-multiply algorithm for fast exponentiation
   def __pow__(self, n):
      if type(n) is not int:
         raise TypeError
      Q = self
      R = self if n & 1 else self.__class__(1)
      i = 2
      while i <= n:
         Q = (Q * Q)
         if n & i == i:
            R = (Q * R)
         i = i << 1
      return R

   # requires the additional % operator (i.e. a Euclidean Domain)
   def powmod(self, n, modulus):
      if type(n) is not int:
         raise TypeError
      Q = self
      R = self if n & 1 else self.__class__(1)
      i = 2
      while i <= n:
         Q = (Q * Q) % modulus
         if n & i == i:
            R = (Q * R) % modulus
         i = i << 1
      return R

# additionally require inverse() on subclasses
class FieldElement(DomainElement):
   def __truediv__(self, other): return self * other.inverse()
   def __rtruediv__(self, other): return self.inverse() * other
   def __div__(self, other): return self.__truediv__(other)
   def __rdiv__(self, other): return self.__rtruediv__(other)

####
## Number theory

# qが素数のベキであればTrueを，それ以外の場合はFalseを返す．
def isPrimePowerQ(q) -> bool:
    divided = False
    for p in sympy.sieve.primerange(2, max(4, (q // 2) + 1)):
        if (q % p) == 0:
            if not divided:
                divided = True
            else:
                return False
    return divided

def PrimePowerQ(q):
    #print(f'PrimePowerQ(q={q})')
    assert(isPrimePowerQ(q))
    for p in sympy.sieve.primerange(2, max(4, (q // 2) + 1)):
        if (q % p) == 0:
            #print(f'PrimePowerQ(q={q}, p={p}, pow={log(q,p)})')
            return p, log(q, p)
    assert(False)
    return 0, 0

# a general Euclidean algorithm for any number type with
# a divmod and a valuation abs() whose minimum value is zero
def gcd(a, b):
   if abs(a) < abs(b):
      return gcd(b, a)
   while abs(b) > 0:
      _,r = divmod(a,b)
      a,b = b,r
   return a

# extendedEuclideanAlgorithm: int, int -> int, int, int
# input (a,b) and output three numbers x,y,d such that ax + by = d = gcd(a,b).
# Works for any number type with a divmod and a valuation abs()
# whose minimum value is zero
def extendedEuclideanAlgorithm(a, b):
   if abs(b) > abs(a):
      (x,y,d) = extendedEuclideanAlgorithm(b, a)
      return (y,x,d)
   if abs(b) == 0:
      return (1, 0, a)
   x1, x2, y1, y2 = 0, 1, 1, 0
   while abs(b) > 0:
      q, r = divmod(a,b)
      x = x2 - q*x1
      y = y2 - q*y1
      a, b, x2, x1, y2, y1 = b, r, x1, x, y1, y
   return (x2, y2, a)

#  q二項係数を計算する
def QBinomial(d, t, q):
    assert (1 <= t <= d)
    if t == 0: return 1
    else: return reduce(operator.mul, [(pow(q, d - i) - 1) / (pow(q, i + 1) - 1) for i in range(t)])

####
## Mod p

# so all IntegersModP are instances of the same base class
class _Modular(FieldElement):
   pass

@memoize
def IntegersModP(p):
   # assume p is prime
   class IntegerModP(_Modular):
      def __init__(self, n):
         try:
            self.n = int(n) % IntegerModP.p
         except:
            raise TypeError("Can't cast type %s to %s in __init__" % (type(n).__name__, type(self).__name__))
         self.field = IntegerModP
      @typecheck
      def __add__(self, other):
         return IntegerModP(self.n + other.n)
      @typecheck
      def __sub__(self, other):
         return IntegerModP(self.n - other.n)
      @typecheck
      def __mul__(self, other):
         return IntegerModP(self.n * other.n)
      def __neg__(self):
         return IntegerModP(-self.n)
      @typecheck
      def __eq__(self, other):
         return isinstance(other, IntegerModP) and self.n == other.n
      @typecheck
      def __ne__(self, other):
         return isinstance(other, IntegerModP) is False or self.n != other.n
      @typecheck
      def __divmod__(self, divisor):
         q,r = divmod(self.n, divisor.n)
         return (IntegerModP(q), IntegerModP(r))
      def inverse(self):
         # need to use the division algorithm *as integers* because we're
         # doing it on the modulus itself (which would otherwise be zero)
         x,y,d = extendedEuclideanAlgorithm(self.n, self.p)
         if d != 1:
            raise Exception("Error: p is not prime in %s!" % (self.__name__))
         return IntegerModP(x)
      def __abs__(self):
         return abs(self.n)
      def __str__(self):
         return str(self.n)
      def __repr__(self):
         return '%d' % (self.n)#'%d (mod %d)' % (self.n, self.p)
      def __pow__(self, n):
         assert type(n) is int and 0 <= n
         return reduce(lambda x, y: x * self, range(1, n+1), IntegerModP(1))
      def __int__(self):
         return self.n
   IntegerModP.p = p
   IntegerModP.__name__ = 'Z/%d' % (p)
   IntegerModP.englishName = 'IntegersMod%d' % (p)
   return IntegerModP

####
## Polynomials

# create a polynomial with coefficients in a field; coefficients are in
# increasing order of monomial degree so that, for example, [1,2,3]
# corresponds to 1 + 2x + 3x^2
@memoize
def polynomialsOver(field=fractions.Fraction):
   class Polynomial(DomainElement):
      operatorPrecedence = 2
      @classmethod
      def factory(cls, L):
         return Polynomial([cls.field(x) for x in L])
      def __init__(self, c):
          # strip all copies of elt from the end of the list
         def strip(L, elt):
            if len(L) == 0: return L
            i = len(L) - 1
            while i >= 0 and L[i] == elt:
                i -= 1
            return L[:i+1]
         if type(c) is Polynomial:
            self.coefficients = c.coefficients
         elif isinstance(c, field):
            self.coefficients = [c]
         elif not hasattr(c, '__iter__') and not hasattr(c, 'iter'):
            self.coefficients = [field(c)]
         else:
            self.coefficients = c
         self.coefficients = strip(self.coefficients, field(0))
      def isZero(self): return self.coefficients == []
      def isOne(self): return self.coefficients == [1]
      def __repr__(self):
         if self.isZero():
            return '0'
         return ' + '.join(['%s x^%d' % (a,i) if i > 0 else '%s'%a
                              for i,a in enumerate(self.coefficients)])
      def __abs__(self): return len(self.coefficients) # the valuation only gives 0 to the zero polynomial, i.e. 1+degree
      def __len__(self): return len(self.coefficients)
      def __sub__(self, other): return self + (-other)
      def __iter__(self): return iter(self.coefficients)
      def __neg__(self): return Polynomial([-a for a in self])
      def iter(self): return self.__iter__()
      def leadingCoefficient(self): return self.coefficients[-1]
      def degree(self): return abs(self) - 1
      @typecheck
      def __eq__(self, other):
         return self.degree() == other.degree() and all([x==y for (x,y) in zip(self, other)])
      @typecheck
      def __ne__(self, other):
          return self.degree() != other.degree() or any([x!=y for (x,y) in zip(self, other)])
      @typecheck
      def __add__(self, other):
         newCoefficients = [sum(x) for x in zip_longest(self, other, fillvalue=self.field(0))]
         return Polynomial(newCoefficients)
      @typecheck
      def __mul__(self, other):
         if self.isZero() or other.isZero():
            return self.Zero()
         newCoeffs = [self.field(0) for _ in range(len(self) + len(other) - 1)]
         for i,a in enumerate(self):
            for j,b in enumerate(other):
               newCoeffs[i+j] += a*b
         return Polynomial(newCoeffs)
      @typecheck
      def __divmod__(self, divisor):
         quotient, remainder = self.Zero(), self
         divisorDeg = divisor.degree()
         divisorLC = divisor.leadingCoefficient()
         while remainder.degree() >= divisorDeg:
            monomialExponent = remainder.degree() - divisorDeg
            monomialZeros = [self.field(0) for _ in range(monomialExponent)]
            monomialDivisor = Polynomial(monomialZeros + [remainder.leadingCoefficient() / divisorLC])
            quotient += monomialDivisor
            remainder -= monomialDivisor * divisor
         return quotient, remainder
      @typecheck
      def __truediv__(self, divisor):
         if divisor.isZero():
            raise ZeroDivisionError
         return divmod(self, divisor)[0]
      @typecheck
      def __mod__(self, divisor):
         if divisor.isZero():
            raise ZeroDivisionError
         return divmod(self, divisor)[1]
      def evaluate(self, v):
         def horner(pol):
            return reduce(lambda x, pair: x + pair[1] * (pol ** pair[0]),
               filter(lambda x: x[1] != 0, enumerate(self.coefficients)), self.Zero())
         if type(v) is int:
            return horner(Polynomial(v))
         elif type(v) is Polynomial:
            return horner(v)
         else:
            assert False
            return None
      # isIrreducible: () -> bool
      # determine if the given monic polynomial with coefficients in Z/p is
      # irreducible over Z/p where p is the given integer
      # Algorithm 4.69 in the Handbook of Applied Cryptography
      def isIrreducible(self):
         if self.coefficients[len(self.coefficients) - 1] != 1:
            raise TypeError('The polynomial is not monic (' + str(self.coefficients) + ')')
         x = self.factory([0,1])
         powerTerm = x
         isUnit = lambda p: p.degree() == 0
         for _ in range(int(self.degree() / 2)):
            powerTerm = powerTerm.powmod(self.field.p, self)
            gcdOverZmodp = gcd(self, powerTerm - x)
            if not isUnit(gcdOverZmodp):
               return False
         return True
      def isPrimitive(self):
         if not self.isIrreducible():
            return False
         def iteratePowersOfPolynomialRoot():
            v = alpha = self.factory([0, 1])
            yield v
            while not v.isOne():
               v = (v * alpha) % self
               yield v
         powers = list(iteratePowersOfPolynomialRoot())
         return len(powers) == (self.field.p ** self.degree() - 1)
      @classmethod
      def Zero(cls):
         return Polynomial([])
      @classmethod
      def One(cls):
         return Polynomial(1)
      @classmethod
      def Alpha(cls):
         return Polynomial([0, 1])
   Polynomial.field = field
   Polynomial.__name__ = '(%s)[x]' % field.__name__
   Polynomial.englishName = 'Polynomials in one variable over %s' % field.__name__
   return Polynomial

# Irreducible Polynomials
# Answers : https://www.jjj.de/mathdata/all-irredpoly.txt
def irreduciblePolynomials(modulus, degree):
   class IrreduciblePolynomialCoefficientsUpdater:
      __path = "poly.xml"
      __singleton = None
      __dictionary = {}
      def __new__(cls, *args, **kwargs):
         if cls.__singleton == None:
            cls.__singleton = super(IrreduciblePolynomialCoefficientsUpdater, cls).__new__(cls)
            try:
               tree = ET.parse(cls.__path)
               for pol in tree.getroot().findall(".//polynomial"):
                  mod = int(pol.attrib.get('modulus'))
                  deg = int(pol.attrib.get('degree'))
                  cls.__addCoef(mod, deg,
                     [int(c.text) for c in sorted(pol.findall("./coefficient"), key=lambda c: int(c.attrib['d']))])
            except:
               print('An error occurs when opening ' + str(cls.__path) + '.')
            # print(f'dictionary in {cls.__path}=\n{cls.__dictionary}')
         return cls.__singleton
      @classmethod
      def __addCoef(cls, mod, deg, coefficients, write=False):
         def create():
            if write == False:
               return
            polynomials = ET.Element("polynomials")
            for (m, dg), coefslist in cls.__dictionary.items():
               for coefs in coefslist:
                  polynomial = ET.Element("polynomial", attrib={'modulus':str(m), 'degree':str(dg)})
                  for d, c in enumerate(coefs):
                     coefficient = ET.Element("coefficient", attrib={'d':str(d)})
                     coefficient.text = str(c)
                     polynomial.append(coefficient)
                  polynomials.append(polynomial)
            root = ET.ElementTree(polynomials)
            root.write(cls.__path, encoding='utf-8')
         if not (mod, deg) in cls.__dictionary.keys():
            cls.__dictionary[(mod, deg)] = [coefficients]
            create()
         elif not coefficients in cls.__dictionary[(mod, deg)]:
            cls.__dictionary[(mod, deg)].append(coefficients)
            create()
      def __init__(self):
         self.Polynomials = self.__dictionary.get((modulus, degree))
         # print(f'IrreduciblePolynomialCoefficientsUpdater(modulus={modulus}, degree={degree})=\n{self.Polynomials}')
      def hasCoef(self):
         return self.Polynomials != None
      def add(self, coefficients):
         self.__addCoef(modulus, degree, coefficients, write=True)
         self.Polynomials = self.__dictionary.get((modulus, degree))
   Zp = IntegersModP(modulus)
   Polynomial = polynomialsOver(Zp)
   CoefficientsUpdater = IrreduciblePolynomialCoefficientsUpdater()
   class IrreduciblePolynomials:
      Modulus = modulus
      Degree = degree
      __Polynomials = []
      def __init__(self, *args, **kwargs):
         self.Count = self.__count()
         if CoefficientsUpdater.hasCoef():
            self.__Polynomials = [ Polynomial.factory(c) for c in CoefficientsUpdater.Polynomials]
            if self.Count != len(self.__Polynomials):
               self.__Polynomials = []
               self.__Interm = enumerate(self.__generateSorted())
            else:
               self.__Interm = None
         else:
            self.__Interm = enumerate(self.__generateSorted())
      def __del__(self):
         list(self.getSorted())
      @typecheck
      def __eq__(self, other):
         return isinstance(other, self.__class__) and self.Modulus == other.modulus and self.Degree == other.Degree
      @typecheck
      def __ne__(self, other): return not self == other
      # The number of monic irreducible polynomials of degree 'degree' over a field with 'modulus' elements.
      # https://www.johndcook.com/blog/2019/03/14/irreducible-polynomials/
      # https://arxiv.org/abs/1001.0409
      def __count(self):
         return sum([sympy.mobius(d) * pow(modulus, degree//d) for d in sympy.divisors(degree)]) // degree
      # __generate: () -> Polynomial
      # generate a random irreducible polynomial of a given degree over Z/p, where p
      # is given by the integer 'modulus'. This algorithm is expected to terminate
      # after 'degree' many irreducilibity tests. By Chernoff bounds the probability
      # it deviates from this by very much is exponentially small.
      def __generate(self):
         while True:
            coefficients = [Zp(random.randint(0, modulus-1)) for _ in range(degree)]
            randomMonicPolynomial = Polynomial(coefficients + [Zp(1)])
            if randomMonicPolynomial.isIrreducible():
               return randomMonicPolynomial
      def __generateEach(self):
         numOfPols = self.__count()
         lst = []
         while len(lst) < numOfPols:
            pol = self.__generate()
            if lst.count(pol) == 0:
               lst.append(pol)
               yield pol
      def __generateSorted(self):
         return sorted(self.__generateEach(),
            key=lambda irr: reduce(lambda cum, c: modulus*cum + c.n, irr.coefficients, 0))
      def get(self, index):
         if self.Count <= index or index < 0:
            return None
         elif index < len(self.__Polynomials):
            return self.__Polynomials[index]
         else:
            assert self.__Interm != None
            for i, pol in self.__Interm:
               self.__Polynomials.append(pol)
               CoefficientsUpdater.add(pol.coefficients)
               if i == index:
                  if i == self.Count - 1:
                     self.__Interm = None
                  return pol
      def getCoefficients(self, index):
         pol = self.get(index)
         return pol.coefficients if pol != None else None
      def getSorted(self):
         return map(self.get, range(self.Count))
   IrreduciblePolynomials.__name__ = 'IrreduciblePolynomials(modulus=' + str(modulus) + ', degree=' + str(degree) + ')'
   return IrreduciblePolynomials

def primitivePolynomials(irreduciblePols):
   class PrimitivePolynomials:
      Modulus = irreduciblePols.Modulus
      Degree = irreduciblePols.Degree
      __Polynomials = []
      def __init__(self, *args, **kwargs):
         self.Count = self.__count()
         self.__Interm = enumerate(self.__generateSorted())
      @typecheck
      def __eq__(self, other):
         return isinstance(other, self.__class__) and self.Modulus == other.modulus and self.Degree == other.Degree
      @typecheck
      def __ne__(self, other): return not self == other
      # http://mathworld.wolfram.com/PrimitivePolynomial.html
      def __count(self):
         return sympy.ntheory.totient(self.Modulus ** self.Degree - 1) // self.Degree
      def __generate(self):
         for irr in self.__generateEach():
            return irr
      def __generateEach(self):
         return filter(lambda irr: irr.isPrimitive(), irreduciblePols.getSorted())
      def __generateSorted(self):
         return self.__generateEach()
      def get(self, index):
         if self.Count <= index or index < 0:
            return None
         elif index < len(self.__Polynomials):
            return self.__Polynomials[index]
         else:
            assert self.__Interm != None
            for i, pol in self.__Interm:
               self.__Polynomials.append(pol)
               if i == index:
                  if i == self.Count - 1:
                     self.__Interm = None
                  return pol
      def getCoefficients(self, index):
         pol = self.get(index)
         return pol.coefficients if pol != None else None
      def getSorted(self):
         return map(self.get, range(self.Count))
   PrimitivePolynomials.__name__ = 'PrimitivePolynomials(modulus=' + str(irreduciblePols.Modulus) + ', degree=' + str(irreduciblePols.Degree) + ')'
   return PrimitivePolynomials

# create a type constructor for the finite field of order p^m for p prime, m >= 1
def finiteField(p, m, polynomialModulus):
   #print(f'finiteFiled(p = {p}, m = {m})')
   Zp = IntegersModP(p)
   if m == 1:
      return Zp
   Polynomial = polynomialsOver(Zp)
   class Fq(FieldElement):
      fieldSize = int(p ** m)
      baseField = Zp
      idealGenerator = polynomialModulus
      operatorPrecedence = 3
      def __init__(self, poly):
         if type(poly) is Fq:
            self.poly = poly.poly
         elif type(poly) is int or type(poly) is Zp:
            self.poly = Polynomial([Zp(poly)])
         elif isinstance(poly, Polynomial):
            self.poly = poly % polynomialModulus
         else:
            self.poly = Polynomial([Zp(x) for x in poly]) % polynomialModulus
         self.filledCoefficients = self.poly.coefficients + [0 for _ in range(polynomialModulus.degree() - self.poly.degree() - 1)]
         self.field = Fq
      def __add__(self, other): return Fq(self.poly + other.poly)
      def __sub__(self, other): return Fq(self.poly - other.poly)
      def __mul__(self, other): return Fq(self.poly * other.poly)
      def __eq__(self, other): return isinstance(other, Fq) and self.poly == other.poly
      def __ne__(self, other): return not self == other
      def __neg__(self): return Fq(-self.poly)
      def __abs__(self): return abs(self.poly)
      def __repr__(self): return repr(self.poly) + ' \u2208 ' + self.__class__.__name__
      def __divmod__(self, divisor):
         q,r = divmod(self.poly, divisor.poly)
         return (Fq(q), Fq(r))
      def inverse(self):
         if self == Fq(0):
            raise ZeroDivisionError
         x,y,d = extendedEuclideanAlgorithm(self.poly, self.idealGenerator)
         if d.degree() != 0:
            raise Exception('Somehow, this element has no inverse! Maybe intialized with a non-prime?')
         return Fq(x) * Fq(d.coefficients[0].inverse())
   Fq.__name__ = 'F_{%d^%d}' % (p,m)
   return Fq

def extendedFiniteField(primitivePolynomial):
   assert primitivePolynomial.isPrimitive()
   p = primitivePolynomial.field.p
   dim = primitivePolynomial.degree()
   field = finiteField(p, dim, primitivePolynomial)
   poly = primitivePolynomial.factory
   class FiniteField:
      Dimension = dim
      P = p
      PrimitivePolynomial = primitivePolynomial
      Zero = field(primitivePolynomial.Zero())
      One = field(primitivePolynomial.One())
      Alpha = field(primitivePolynomial.Alpha())
      def __init__(self):
         self.Members = self.__extend()
         self.MemberCoefficients = [m.filledCoefficients for m in self.Members]
      def __extend(self):
         v = self.Alpha
         polynomials = [self.Zero, self.One]
         while v != self.One:
               polynomials.append(v)
               v = v * self.Alpha
         return polynomials
      def __eq__(self, other):
         return isinstance(other, self.__class__) and self.Dimension == other.Dimension and self.P == other.P and self.PrimitivePolynomial == other.PrimitivePolynomial
      def __ne__(self, other):
         return not self == other
      def tracefun(self):
         def coefficients(n, q):
               #print(f'coefficients(n={n},q={q})')
               return sum([[0 for _ in range(q ** i - (q ** (i - 1) + 1 if i > 0 else 0))] + [1] for i in range(0, n)], [])
         coeffs = coefficients(primitivePolynomial.degree(), p)
         #print(f'tracefun={coeffs}')
         return poly(coeffs)
   FiniteField.__name__ = 'Field(' + str(p) + '^' + str(dim) + ' [' + str(primitivePolynomial) + '])'
   return FiniteField

####
## Examples
# if __name__ == "__main__":
   #print(f'PrimePowerQ(100)={PrimePowerQ(100)}')
   #print(f'PrimePowerQ(125)={PrimePowerQ(125)}')
   #print(f'PrimePowerQ(73*73)={PrimePowerQ(73*73)}')
   #print(f'QBinomial(4,2,0.5)={QBinomial(4,2,0.5)}') # 35/16
   #print(f'QBinomial(4,2,5)={QBinomial(4,2,5)}') # 806
   #print(f'I_q(8, 2)={I_q(8, 2)}') # 30
   #print(f'I_q(8, 256)={I_q(8, 256)}') # 2,305,843,008,676,823,040
