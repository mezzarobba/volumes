r"""
Volumes of semi-algebraic sets

This is a *partial* implementation in SageMath of the algorithm for computing
volumes of basic semi-algebraic sets to high precision described in

    [LMS2019] Pierre Lairez, Marc Mezzarobba, and Mohab Safey El Din. 2019.
    Computing the volume of compact semi-algebraic sets. ISSAC ’19.
    https://doi.org/10.1145/3326229.

The present code is currently limited to semi-algebraic sets defined by a
single inequality.

In addition, it computes Picard-Fuchs equations using Chyzak's general Creative
Telescoping algorithm instead of the reduction-based procedure mentioned in the
article. Besides being slower for moderately complicated examples, this means
that the present implementation can fail, or even yield incorrect results, if
the certificates in the telescoping identity found by Chyzak's algorithm have
spurious poles. There is currently **no reliable check** for this in the code.

Finally, there are some limitations in the supported geometries.

(See also the LIMITATIONS section below.)


REQUIREMENTS:

SageMath >= 9.8 with the optional packages `ore_algebra` and `msolve`.


AUTHORS:

- Marc Mezzarobba (2019, 2023)
- Mohab Safey El Din (2019, 2023)


EXAMPLES::

    sage: from volume import volume1

Henrion, Lasserre, Savorgnan 2008, Section 4.2::

    sage: P.<x1,x2> = QQ[]
    sage: f = x1*(x1^2+x2^2)-(x1^4+x1^2*x2^2+x2^4)
    sage: volume1(-f, 100)
    [1.05804962913662707951321...]

Henrion, Lasserre, Savorgnan 2008, Section 4.3::

    sage: P.<x1,x2> = QQ[]
    sage: f = -(x1^2+x2^2)^3 + 4*x1^2*x2^2 # folium
    sage: volume1(-f, 100)
    [1.5707963267948966192...]

Tacchi, Lasserre, Henrion, 2022, Section 6.5.2::

    sage: P.<x1,x2> = QQ[]
    sage: f = -(1/16 - (x1-1/2)^2 - x2^2)*((x1+1/2)^2 + x2^2 - 1/16)
    sage: volume1(f, 100)
    [0.3926990816987241548078...]

A torus (cf. [LMS2019] Section 1])::

    sage: P.<x,y,z> = QQ[]
    sage: volume1((x^2+y^2+z^2+3)^2 - 16 * (x^2+y^2), 100)
    [39.478417604357434475...]

A nice-looking non-convex set::

    sage: P.<x,y,z> = QQ[]
    sage: volume1(x^2+y^2+z^2-(1 - 2^10*(x^2*y^2+y^2*z^2+z^2*x^2)), 100)
    [0.1085754214603609377395...]

A bretzel (TODO: check output)::

    sage: P.<x,y,z> = QQ[]
    sage: volume1((x^2*(1-x^2)-y^2)^2 + z^2-1/100*(x^2*(1-x^2)-y^2)^2 + z^2-1/100, 100)
    [0.14000962837928496908666...]

A Goursat surface (TODO: check output)::

    sage: P.<x,y,z> = QQ[]
    sage: volume1(x^4+y^4+z^4 - (x^2+y^2+z^2), 100)
    [10.4760907538928054...]

LIMITATIONS:

Unsupported geometries::

    sage: volume1((x^2+y^2+z^2-1)*((x-4)^2+y^2+z^2-1)+1/2^64, 100)
    Traceback (most recent call last):
    ...
    ValueError: positive-dimensional ideal

We cannot compute the volume of the unit ball of ℝ^4 (!)::

    sage: from volume import volume1
    sage: P.<w,x,y,z> = QQ[]
    sage: volume1(w^2+x^2+y^2+z^2-1, 100)
    Traceback (most recent call last):
    ...
    AssertionError: trivial telescoper

This is related to the fact that one of the rational functions that the
algorithm integrates can be decomposed as a sum of derivatives::

    6*x^2/(w^2+x^2+y^2+z^2-1) = diff(2*x^3/(w^2+x^2+y^2+z^2-1),x)
                                +diff(2*x^4*y/(w^2+x^2-1)/(w^2+x^2+y^2+z^2-1),y)
                                +diff(2*z*x^4/(w^2+x^2+y^2+z^2-1)/(w^2+x^2-1),z)
"""

import logging
from itertools import count, pairwise
from sage.all import *
from sage.rings.real_arb import RealBall
from ore_algebra import OreAlgebra


logger = logging.getLogger("volume")


def picard_fuchs_equation(pol):
    # TODO: use Lairez' algorithm instead
    logger.info("computing Picard-Fuchs equation for %s...", pol)
    pol = pol.numerator().change_ring(ZZ)
    Pol = pol.parent()
    x = Pol.gens()
    Alg, Dx = OreAlgebra(Frac(Pol), names=['D' + repr(xi) for xi in x]).objgens()
    integrand = x[1]*pol.derivative(x[1])/pol
    id = Alg.ideal(
        integrand*Dxi - integrand.derivative(xi)
        for xi, Dxi in zip(x, Dx))
    for Dxi in reversed(Dx[1:]):
        logger.info("  eliminating %s", Dxi)
        tel = id.ct(Dxi, certificates=False)
        assert not tel[0].is_one(), "trivial telescoper"
        id = tel[0].parent().ideal(tel)
    assert len(tel) == 1
    pf = tel[0].numerator()
    logger.info("done, order = %s, degree = %s", pf.order(), pf.degree())
    return pf


def critical_values(pol):
    Pol = pol.parent()
    x = Pol.gens()
    syst = [pol] + [pol.derivative(xi) for xi in x[1:]]
    id = Pol.ideal(syst)
    variety = id.variety(AA, algorithm="msolve", proof=False)
    # TODO: There may be several critical points above a given critical value.
    # At the moment, we let QQbar take care of deciding if two critical values
    # (defined as projections of critical points) are equal. To do better, we
    # could isolate the critical values (as opposed to just the critical
    # points) by computing the squarefree part of the resultant of the
    # eliminating polynomial with the bivariate polynomial that defines the
    # first coordinate of critical points. (Cf. old version of volume1.)
    return list(sorted({point[x[0]] for point in variety}))


def dsolve(op, ini, alpha, prec):

    R = RealBallField(prec)

    mon = op.local_basis_monomials(alpha)
    try:
        j0 = mon.index(1)
    except ValueError:
        return R.zero()

    u_prev = alpha
    delta = identity_matrix(R, op.order())
    rows = []
    for u, j, _ in ini:
        assert u_prev < u
        tm = op.numerical_transition_matrix([u_prev, u], RBF(1) >> prec,
                                            assume_analytic=True, squash_intervals=True)
        delta = tm*delta
        rows.append(delta.row(j))
        u_prev = u

    mat = matrix(rows)
    rhs = vector([s for _, _, s in ini])
    inv = ~mat
    assert not inv.is_one()
    sol = inv*rhs

    res = sol[j0]
    if not isinstance(res, RealBall):
        assert res.imag().contains_zero()
        res = res.real()
    assert not res > 0
    return res


class PrecisionError(Exception):
    pass


def univariate_volume(pol, prec): # Theta_U = {pol <= 0}
    myRIF = RealIntervalField(prec)
    myRBF = RealBallField(prec)
    rts = pol.roots(myRIF, multiplicities=False)
    rts = [myRBF(rt) for rt in rts]
    length = myRBF.zero()
    for a, b in zip(rts[:-1], rts[1:]):
        m = (a + b)/2
        val = pol(m)
        if val <= 0:
            length += b - a
        elif val >= 0:
            pass
        else:
            raise PrecisionError
    return length


def random_rational_points(a, b, order, degree):
    aa = RBF.coerce(a)
    bb = RBF.coerce(b)
    delta = bb - aa
    rng = (abs(RR(delta).log(2)) + 1).floor() * 8 * (order+1)**2 * (degree + 1)**2
    for k in count(4):
        l = (rng/k).floor()
        h = rng - l
        if h - l >= (degree + 1)*order:
            break
    randints = {ZZ.random_element(l, h + 1) for _ in range(order)}
    while len(randints) < order:
        randints = {ZZ.random_element(1, rng + 1) for _ in range(order)}
    return [RIF(aa + RBF(j).add_error(0.5)*delta/(rng + 1)).simplest_rational()
            for j in sorted(randints)]


def volume1(f, prec=53, *, depth=0):
    r"""
    Compute the volume of a compact semi-algebraic set defined by a single
    inequality.

    INPUT:

    - ``f``: polynomial with rational coefficients
    - ``prec`` (optional, default 53): initial working precision in bits

    OUTPUT:

    A real interval containing the volume of the set ``{ f ≤ 0 }``. This set is
    assumed to be compact; otherwise, the function may silently return
    meaningless results.

    ALGORITHM: [LMS2019], Algorithm 1.

    EXAMPLES::

        sage: from volume import volume1
        sage: P.<x,y> = QQ[]
        sage: volume1(x^2 + y^2 - 1)
        [3.141592653589... +/- ...]
        sage: volume1(x^2 + 2*y^2 - 1, 100)  # π√2/2
        [2.221441469079183123507940495...]

    TESTS::

        sage: P.<x> = QQ[]
        sage: volume1(x^2 - 2)
        [2.82842712474619...]
        sage: volume1(x^2 + 1)
        0
    """

    if f.parent().ngens() == 1:
        vol = univariate_volume(f, prec)
        logger.info("%s slice length = %s", "·"*(depth*2 + 1), vol)
        return vol

    crit = critical_values(f)
    assert crit == list(sorted(crit))
    if not crit:
        return None # empty slice

    x0 = f.parent().gen(0)

    pf = picard_fuchs_equation(f)
    deg = pf.leading_coefficient().degree()

    logger.info("%s PF eq order = %s",
                "·"*(depth*2+1), pf.order())
    Dt = pf.parent().gen()
    intpf = pf*Dt
    R = RealBallField(prec)
    vol = R.zero()

    for a0, a1 in pairwise(crit):
        assert a0 < a1
        logger.info("%s interval: [%s, %s]", "·"*(depth*2+1), a0, a1)
        ini = []
        slices = random_rational_points(a0, a1, pf.order(), deg)
        for s, rho in enumerate(slices):
            newf = f.polynomial(x0)(rho)
            slice_volume = volume1(newf, prec, depth=depth+1)
            if slice_volume is None:
                continue
            logger.info("%s slice #%s: ρ = %s", "·"*(depth*2+2), s + 1, rho)
            slice_volume = volume1(newf, prec, depth=depth+1)
            ini.append([rho, 1, slice_volume])
        logger.info("%s integrating PF equation over [%s, %s]...",
                    "·"*(depth*2+2), a0, a1)
        k = intpf.local_basis_monomials(a1).index(1)
        ini.append([a1, k, 0])
        pvol = -dsolve(intpf, ini, a0, prec)
        assert not pvol < 0
        logger.info("%s ...piece volume = %s", "·"*(depth*2+2), pvol)
        vol += pvol

    if depth > 0:
        logger.info("%s slice volume = %s", "·"*(depth*2 + 1), vol)
    return vol
