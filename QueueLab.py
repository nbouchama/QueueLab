#!/usr/bin/env python3
"""
Files d'attente — Laboratoire de simulation (Tkinter, bibliothèque standard uniquement)

Simule des systèmes de files d'attente paramétrables et compare la simulation à la théorie.

MODÈLES (notation de Kendall  A/B/c/K/N/discipline)
  - Lois d'arrivée et de service : M (exponentielle), D (déterministe), Eₖ (Erlang), H₂ (hyperexponentielle),
    U (uniforme), LN (log-normale), G (générale, cv² réglable).
  - 1 à 10 serveurs ou une infinité ; capacité K finie ou infinie ; population N finie ou infinie.
  - Disciplines FIFO, LIFO, SIRO, SJF ; abandons (patience) ; refus aléatoires (balking).
  - 23 modèles prêts à l'emploi : M/M/1, M/M/1/K, M/M/c, Erlang-B, M/M/∞, M/D/1, D/M/1, D/D/1/K, M/Eₖ/1,
    M/H₂/1, M/G/1, M/G/1/K, G/G/1, G/G/c, population finie, abandons, refus, LIFO, SIRO, SJF…

ARCHITECTURE DU FICHIER
  1. MOTEUR      : lois de probabilité, simulation à événements discrets (classe Sim), réplications
                   et intervalles de confiance, formules analytiques (fonction theorie).
  2. KIT GRAPHIQUE : thème, icônes dessinées, info-bulles, widgets mobiles (boutons, curseurs, puces, cartes).
  3. INTERFACE   : classe App avec 5 écrans (Modèle, Simuler, Résultats, Études, Aide).

THÉORIE UTILISÉE
  - Naissance et mort (exact)  : M/M/c/K/N avec abandons exponentiels et refus aléatoires.
  - Pollaczek-Khinchine (exact): M/G/1.        - G/M/1 (exact) : arrivées D, Eₖ, H₂, U.
  - M/G/1/K (exact)            : chaîne de Markov incluse, service M, D, Eₖ ou H₂.
  - Kingman / Allen-Cunneen    : approximations G/G/1 et G/G/c.   - D/D/c : formules déterministes.
  Le moteur a été validé contre ces formules (écarts compatibles avec les IC à 95 %).

Lancement : python files_attente.py
"""
import colorsys
import csv
import heapq
import itertools
import math
import random
import time
import tkinter as tk
import tkinter.font as tkfont
from bisect import bisect_right, insort
from collections import deque
from tkinter import filedialog, messagebox

# ======================================================================
#  MOTEUR : lois, simulation à événements discrets, théorie des files
# ======================================================================
INF = float("inf")

DIST_LIB = {
    "M": "Exponentielle (Markov)",
    "D": "Déterministe (constante)",
    "E": "Erlang-k (régulière)",
    "H": "Hyperexponentielle H₂ (irrégulière)",
    "U": "Uniforme sur [0 ; 2·moyenne]",
    "LN": "Log-normale",
    "G": "Générale (log-normale, cv² réglable)",
}
DIST_COURT = {"M": "M", "D": "D", "E": "Eₖ", "H": "H₂", "U": "U", "LN": "LN", "G": "G"}
DISCIPLINES = {
    "FIFO": "FIFO : premier arrivé, premier servi",
    "LIFO": "LIFO : dernier arrivé, premier servi",
    "SIRO": "SIRO : service dans un ordre aléatoire",
    "SJF": "SJF : le plus court service d'abord",
}

DEFAUT = dict(arr="M", lam=0.8, arr_forme=2.0, srv="M", mu=1.0, srv_forme=2.0,
              c=1, K=INF, N=INF, disc="FIFO", patience=None, balk=0.0,
              horizon=2000.0, warmup=0.0, seed=0)


def scv(code, forme):
    """Carré du coefficient de variation d'une loi."""
    if code == "M":
        return 1.0
    if code == "D":
        return 0.0
    if code == "E":
        return 1.0 / max(1, int(round(forme)))
    if code == "U":
        return 1.0 / 3.0
    if code == "H":
        return max(forme, 1.0001)
    return max(forme, 1e-6)


def sampler(code, mean, forme, rng):
    """Renvoie une fonction sans argument qui tire une durée de moyenne `mean`."""
    import math
    if code == "M":
        r = 1.0 / mean
        return lambda: rng.expovariate(r)
    if code == "D":
        return lambda: mean
    if code == "E":
        k = max(1, int(round(forme)))
        return lambda: rng.gammavariate(k, mean / k)
    if code == "U":
        return lambda: rng.uniform(0.0, 2.0 * mean)
    if code == "H":
        c2 = max(forme, 1.0001)
        p = 0.5 * (1 + math.sqrt((c2 - 1) / (c2 + 1)))
        r1, r2 = 2 * p / mean, 2 * (1 - p) / mean
        return lambda: rng.expovariate(r1 if rng.random() < p else r2)
    c2 = max(forme, 1e-6)
    s2 = math.log(1 + c2)
    mu = math.log(mean) - s2 / 2
    sg = math.sqrt(s2)
    return lambda: rng.lognormvariate(mu, sg)


def lettre(code, forme):
    if code == "E":
        return f"E{int(round(forme))}"
    return DIST_COURT[code] if code != "E" else "E"


def kendall(p):
    c = "∞" if p["c"] == INF else str(p["c"])
    s = f"{lettre(p['arr'], p['arr_forme'])}/{lettre(p['srv'], p['srv_forme'])}/{c}"
    k_txt = "∞" if p["K"] == INF else str(p["K"])
    n_txt = "∞" if p["N"] == INF else str(p["N"])
    if p["K"] != INF or p["N"] != INF or p["disc"] != "FIFO":
        s += "/" + k_txt
    if p["N"] != INF or p["disc"] != "FIFO":
        s += "/" + n_txt
    if p["disc"] != "FIFO":
        s += "/" + p["disc"]
    return s


def normaliser(p):
    """Copie propre et cohérente des paramètres."""
    q = dict(DEFAUT)
    q.update(p)
    if q["K"] != INF:
        q["K"] = max(1, int(q["K"]))
    if q["c"] != INF:
        q["c"] = max(1, int(q["c"]))
        if q["K"] != INF:
            q["c"] = min(q["c"], q["K"])
    if q["N"] != INF:
        q["N"] = max(1, int(q["N"]))
    q["balk"] = min(0.95, max(0.0, q["balk"]))
    return q


def charge(p):
    """Charge offerte rho = lambda / (c * mu) (population infinie)."""
    lam = p["lam"] * (1 - p["balk"])
    if p["N"] != INF:
        return None
    return lam / (p["mu"] * p["c"]) if p["c"] != INF else 0.0


# ----------------------------------------------------------------------
#  Simulation
# ----------------------------------------------------------------------
import heapq
import itertools
import math
import random
from bisect import bisect_right, insort
from collections import deque


class Client:
    __slots__ = ("id", "t_arr", "svc", "t_start", "slot", "src", "etat", "t_pat")

    def __init__(self, cid, t, svc, src):
        self.id, self.t_arr, self.svc, self.src = cid, t, svc, src
        self.t_start = None
        self.slot = None
        self.etat = "q"
        self.t_pat = None


class Sim:
    """Simulation à événements discrets d'une file G/G/c/K/N (+ abandons, refus).

    step() traite UN événement et renvoie la liste des (t, nature, id, info) produits,
    ou None si la simulation est terminée (horizon atteint)."""

    def __init__(self, params, seed=None, record=True):
        p = self.p = normaliser(params)
        self.seed = seed or p["seed"] or random.randrange(1, 10 ** 9)
        s = self.seed
        self.r_arr, self.r_srv, self.r_misc = (random.Random(s * 3 + 1), random.Random(s * 3 + 2),
                                               random.Random(s * 3 + 3))
        self.s_arr = sampler(p["arr"], 1.0 / p["lam"], p["arr_forme"], self.r_arr)
        self.s_srv = sampler(p["srv"], 1.0 / p["mu"], p["srv_forme"], self.r_srv)
        self.c, self.K, self.N = p["c"], p["K"], p["N"]
        self.record = record
        self.t = 0.0
        self.heap = []
        self.seq = itertools.count()
        self.queue = deque()
        self.busy = {}
        self.free = list(range(int(self.c))) if self.c != INF else []
        self.pool_next = 0
        self.nid = 1
        self.traj_t, self.traj_n, self.skip, self.cnt = [0.0], [0], 1, 0
        self.warm_done = p["warmup"] <= 0
        self.reset_stats(0.0)
        if self.N == INF:
            self._planifier(None)
        else:
            for i in range(self.N):
                self._planifier(i)

    # --- statistiques
    def reset_stats(self, t0):
        self.t0 = self.tl = t0
        self.a_n = self.a_q = self.a_b = 0.0
        self.tn = {}
        self.arrivals = self.accepted = self.blocked = self.balked = 0
        self.served = self.reneged = self.n_wq = self.n_leave = 0
        self.wq_sum = self.soj_sum = self.svc_sum = 0.0
        self.wmax = 0.0
        self.waits = []
        self.n_seen = 0

    def _integ(self, t):
        dt = t - self.tl
        if dt > 0:
            nq = len(self.queue)
            nb = len(self.busy)
            self.a_n += (nq + nb) * dt
            self.a_q += nq * dt
            self.a_b += nb * dt
            self.tn[nq + nb] = self.tn.get(nq + nb, 0.0) + dt
            self.tl = t

    def n(self):
        return len(self.queue) + len(self.busy)

    def metrics(self, t=None):
        if t is not None:
            self._integ(t)
        T = max(self.tl - self.t0, 1e-12)
        nan = float("nan")
        busy = self.a_b / T
        return dict(
            L=self.a_n / T, Lq=self.a_q / T, busy=busy,
            util=busy / self.c if self.c != INF else busy,
            W=self.soj_sum / self.n_leave if self.n_leave else nan,
            Wq=self.wq_sum / self.n_wq if self.n_wq else nan,
            lam_eff=self.accepted / T, thr=self.served / T,
            p_block=self.blocked / self.arrivals if self.arrivals else nan,
            p_balk=self.balked / self.arrivals if self.arrivals else nan,
            p_renege=self.reneged / self.accepted if self.accepted else nan,
            wmax=self.wmax, served=self.served, arrivals=self.arrivals,
            duree=T)

    def pn(self):
        T = max(self.tl - self.t0, 1e-12)
        if not self.tn:
            return []
        return [self.tn.get(i, 0.0) / T for i in range(max(self.tn) + 1)]

    # --- planification
    def _push(self, t, kind, data):
        heapq.heappush(self.heap, (t, next(self.seq), kind, data))

    def _planifier(self, src):
        self._push(self.t + self.s_arr(), "ARR", src)

    def next_time(self):
        return self.heap[0][0] if self.heap else INF

    def finished(self):
        return (not self.heap) or self.heap[0][0] > self.p["horizon"]

    def _tracer(self):
        if not self.record:
            return
        self.cnt += 1
        if self.cnt % self.skip:
            return
        self.traj_t.append(self.t)
        self.traj_n.append(self.n())
        if len(self.traj_t) > 60000:
            self.traj_t, self.traj_n = self.traj_t[::2], self.traj_n[::2]
            self.skip *= 2

    # --- coeur
    def step(self):
        if self.finished():
            return None
        te, _, kind, data = heapq.heappop(self.heap)
        if not self.warm_done and te >= self.p["warmup"]:
            self._integ(self.p["warmup"])
            self.reset_stats(self.p["warmup"])
            self.warm_done = True
        self._integ(te)
        self.t = te
        out = []
        if kind == "ARR":
            self._arrivee(data, out)
        elif kind == "DEP":
            self._depart(data, out)
        else:
            self._abandon(data, out)
        self._tracer()
        return out

    def advance_to(self, t_target):
        """Traite tous les événements jusqu'à t_target ; renvoie la liste des messages."""
        out = []
        while not self.finished() and self.heap[0][0] <= t_target:
            r = self.step()
            if r:
                out.extend(r)
        return out

    def close(self):
        self._integ(min(self.p["horizon"], max(self.t, self.p["horizon"])))

    def _retour_source(self, src):
        if self.N != INF:
            self._planifier(src)

    def _arrivee(self, src, out):
        t = self.t
        self.arrivals += 1
        if self.N == INF:
            self._planifier(None)
        cid = self.nid
        self.nid += 1
        n = self.n()
        if self.p["balk"] > 0 and self.r_misc.random() < self.p["balk"]:
            self.balked += 1
            out.append((t, "balk", cid, n))
            self._retour_source(src)
            return
        if n >= self.K:
            self.blocked += 1
            out.append((t, "block", cid, n))
            self._retour_source(src)
            return
        self.accepted += 1
        cl = Client(cid, t, self.s_srv(), src)
        if len(self.busy) < self.c:
            out.append((t, "arr", cid, (n, 0)))
            self._demarrer(cl, out)
        else:
            self.queue.append(cl)
            out.append((t, "arr", cid, (n, len(self.queue))))
            if self.p["patience"]:
                cl.t_pat = t + self.r_misc.expovariate(1.0 / self.p["patience"])
                self._push(cl.t_pat, "REN", cl)

    def _demarrer(self, cl, out):
        t = self.t
        cl.etat = "s"
        cl.t_start = t
        w = t - cl.t_arr
        self.wq_sum += w
        self.n_wq += 1
        if w > self.wmax:
            self.wmax = w
        if self.record:
            self.n_seen += 1
            if len(self.waits) < 50000:
                self.waits.append(w)
            else:
                j = self.r_misc.randrange(self.n_seen)
                if j < 50000:
                    self.waits[j] = w
        if self.c != INF:
            cl.slot = self.free.pop(0)
        else:
            cl.slot = self.pool_next
            self.pool_next += 1
        self.busy[cl.id] = cl
        self._push(t + cl.svc, "DEP", cl)
        out.append((t, "start", cl.id, (cl.slot, w, cl.svc)))

    def _choisir(self):
        d = self.p["disc"]
        q = self.queue
        if d == "FIFO":
            return q.popleft()
        if d == "LIFO":
            return q.pop()
        if d == "SIRO":
            i = self.r_misc.randrange(len(q))
        else:
            i = min(range(len(q)), key=lambda k: q[k].svc)
        cl = q[i]
        del q[i]
        return cl

    def _depart(self, cl, out):
        t = self.t
        del self.busy[cl.id]
        if self.c != INF:
            insort(self.free, cl.slot)
        cl.etat = "gone"
        self.soj_sum += t - cl.t_arr
        self.n_leave += 1
        self.served += 1
        self.svc_sum += cl.svc
        out.append((t, "dep", cl.id, (cl.slot, t - cl.t_arr)))
        self._retour_source(cl.src)
        if self.queue:
            self._demarrer(self._choisir(), out)

    def _abandon(self, cl, out):
        if cl.etat != "q":
            return
        t = self.t
        self.queue.remove(cl)
        cl.etat = "gone"
        w = t - cl.t_arr
        self.reneged += 1
        self.wq_sum += w
        self.n_wq += 1
        self.soj_sum += w
        self.n_leave += 1
        out.append((t, "renege", cl.id, w))
        self._retour_source(cl.src)


def lancer(params, seed=None):
    """Simulation complète sans animation ; renvoie (métriques, pn, attentes)."""
    s = Sim(params, seed, record=True)
    while s.step() is not None:
        pass
    s.close()
    return s.metrics(), s.pn(), s.waits


# ----------------------------------------------------------------------
#  Agrégation de réplications
# ----------------------------------------------------------------------
T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
        2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
        2.052, 2.048, 2.045, 2.042]


def agreger(liste):
    """liste de dict de métriques -> {clé: (moyenne, demi-largeur IC95)}"""
    out = {}
    for k in liste[0]:
        v = [m[k] for m in liste if m[k] == m[k]]
        if not v:
            out[k] = (float("nan"), 0.0)
            continue
        mu = sum(v) / len(v)
        if len(v) > 1:
            sd = math.sqrt(sum((x - mu) ** 2 for x in v) / (len(v) - 1))
            t = T975[len(v) - 2] if len(v) - 1 <= 30 else 1.96
            out[k] = (mu, t * sd / math.sqrt(len(v)))
        else:
            out[k] = (mu, 0.0)
    return out


# ----------------------------------------------------------------------
#  Théorie
# ----------------------------------------------------------------------
def _fin(L, Lq, lam_eff, ES, c, pb, pbalk, pren, nature, note, pn=None, thr=None):
    busy = lam_eff * ES if thr is None else thr * ES
    return dict(nature=nature, note=note, L=L, Lq=Lq,
                W=L / lam_eff if lam_eff > 0 else None,
                Wq=Lq / lam_eff if lam_eff > 0 else None,
                busy=busy, util=busy / c if c != INF else busy,
                lam_eff=lam_eff, thr=lam_eff if thr is None else thr,
                p_block=pb, p_balk=pbalk, p_renege=pren, pn=pn)


def _rien(note):
    return dict(nature=None, note=note, pn=None, L=None, Lq=None, W=None, Wq=None, busy=None, util=None,
                lam_eff=None, thr=None, p_block=None, p_balk=None, p_renege=None)


def _naissance_mort(p):
    lam, mu, c, K, N = p["lam"], p["mu"], p["c"], p["K"], p["N"]
    q = p["balk"]
    th = 1.0 / p["patience"] if p["patience"] else 0.0
    Ke = min(K, N)
    if Ke == INF and th == 0 and c != INF and N == INF and lam * (1 - q) >= c * mu:
        return _rien("Système instable (ρ ≥ 1) : la file grossit sans limite, pas de régime stationnaire.")

    def lraw(n):
        return lam if N == INF else (N - n) * lam

    w = [1.0]
    tot = 1.0
    n = 0
    lim = Ke if Ke != INF else 30000
    while n < lim:
        mun = min(n + 1, c) * mu + max(n + 1 - c, 0) * th
        w.append(w[-1] * lraw(n) * (1 - q) / mun)
        tot += w[-1]
        n += 1
        if Ke == INF and n > c + 5 and w[-1] < 1e-17 * tot:
            break
    pn = [x / tot for x in w]
    L = sum(i * x for i, x in enumerate(pn))
    Lq = sum(max(i - c, 0) * x for i, x in enumerate(pn))
    busy = sum(min(i, c) * x for i, x in enumerate(pn))
    lam_adm = sum(lraw(i) * (1 - q) * x for i, x in enumerate(pn) if i < Ke)
    offered = lam if N == INF else sum(lraw(i) * x for i, x in enumerate(pn))
    blocked = (1 - q) * lraw(K) * pn[K] if (K < N and K < INF and K < len(pn)) else 0.0
    thr = sum(min(i, c) * mu * x for i, x in enumerate(pn))
    ren = sum(max(i - c, 0) * th * x for i, x in enumerate(pn))
    ES = 1.0 / mu
    r = dict(nature="exacte", note="Processus de naissance et de mort : probabilités d'état exactes.",
             L=L, Lq=Lq, W=L / lam_adm if lam_adm > 0 else None, Wq=Lq / lam_adm if lam_adm > 0 else None,
             busy=busy, util=busy / c if c != INF else busy, lam_eff=lam_adm, thr=thr,
             p_block=blocked / offered if offered > 0 else 0.0, p_balk=q,
             p_renege=ren / lam_adm if lam_adm > 0 else 0.0, pn=pn)
    return r


def _gm1_sigma(p):
    """Racine sigma de G/M/1 (exact) pour A in D,E,H,U,M."""
    lam, mu = p["lam"], p["mu"]
    m = 1.0 / lam
    A = p["arr"]
    k = max(1, int(round(p["arr_forme"])))
    if A == "H":
        c2 = max(p["arr_forme"], 1.0001)
        pp = 0.5 * (1 + math.sqrt((c2 - 1) / (c2 + 1)))
        m1, m2 = m / (2 * pp), m / (2 * (1 - pp))

    def lst(s):
        if A == "M":
            return lam / (lam + s)
        if A == "D":
            return math.exp(-s * m)
        if A == "E":
            return (k / m / (k / m + s)) ** k
        if A == "U":
            return 1.0 if s < 1e-12 else (1 - math.exp(-2 * m * s)) / (2 * m * s)
        return pp / (1 + s * m1) + (1 - pp) / (1 + s * m2)
    sg = 0.0
    for _ in range(20000):
        nv = lst(mu * (1 - sg))
        if abs(nv - sg) < 1e-14:
            break
        sg = nv
    return sg


def _poisson_a(lam, B, mu, forme, n):
    """P(k arrivées Poisson pendant un service) pour B in M, D, E, H."""
    a = []
    if B == "M":
        r = mu / (lam + mu)
        for k in range(n):
            a.append(r * (1 - r) ** k)
    elif B == "D":
        s = lam / mu
        x = math.exp(-s)
        for k in range(n):
            a.append(x)
            x *= s / (k + 1)
    elif B == "E":
        kk = max(1, int(round(forme)))
        r = kk * mu / (kk * mu + lam)
        x = r ** kk
        for j in range(n):
            a.append(x)
            x *= (j + kk) / (j + 1) * (1 - r)
    else:
        c2 = max(forme, 1.0001)
        pp = 0.5 * (1 + math.sqrt((c2 - 1) / (c2 + 1)))
        m = 1.0 / mu
        m1, m2 = m / (2 * pp), m / (2 * (1 - pp))
        for k in range(n):
            g1 = (1 / (1 + lam * m1)) * (lam * m1 / (1 + lam * m1)) ** k
            g2 = (1 / (1 + lam * m2)) * (lam * m2 / (1 + lam * m2)) ** k
            a.append(pp * g1 + (1 - pp) * g2)
    return a


def _mg1k(p):
    lam = p["lam"] * (1 - p["balk"])
    mu, K = p["mu"], p["K"]
    a = _poisson_a(lam, p["srv"], mu, p["srv_forme"], max(K, 2))
    if a[0] <= 1e-300:
        return None
    pi = [1.0]
    for j in range(K - 1):
        s = pi[j] - pi[0] * a[j] - sum(pi[i] * a[j - i + 1] for i in range(1, j + 1))
        pi.append(s / a[0])
    if any((x != x) or x < -1e-9 or x > 1e12 for x in pi):
        return None
    tot = sum(pi)
    pi = [max(x, 0.0) / tot for x in pi]
    ES = 1.0 / mu
    rho = lam * ES
    den = pi[0] + rho
    pn = [x / den for x in pi] + [1 - 1 / den]
    L = sum(i * x for i, x in enumerate(pn))
    lam_eff = lam * (1 - pn[K])
    Lq = L - lam_eff * ES
    r = _fin(L, max(Lq, 0.0), lam_eff, ES, 1, (1 - p["balk"]) * pn[K], p["balk"], 0.0, "exacte",
             "M/G/1/K résolue par la chaîne de Markov incluse aux instants de départ.", pn)
    return r


def theorie(p):
    """Solution analytique quand elle existe. nature : 'exacte', 'approchée' ou None."""
    p = normaliser(p)
    lam, mu, c, K, N, q = p["lam"], p["mu"], p["c"], p["K"], p["N"], p["balk"]
    A, B = p["arr"], p["srv"]
    ca2, cs2 = scv(A, p["arr_forme"]), scv(B, p["srv_forme"])
    ES = 1.0 / mu
    th = p["patience"]
    if p["disc"] == "SJF":
        return _rien("Discipline SJF : aucune formule simple ici (la simulation fait foi).")
    if A == "M" and B == "M":
        return _naissance_mort(p)
    if N != INF:
        return _rien("Population finie non markovienne : simulation seule.")
    if th:
        return _rien("Abandons avec lois non exponentielles : simulation seule.")
    if A == "D" and B == "D":
        if q > 0:
            return _rien("D/D avec refus aléatoires : simulation seule.")
        cap = c * mu if c != INF else INF
        if lam <= cap:
            return _fin(lam * ES, 0.0, lam, ES, c, 0.0, 0.0, 0.0, "exacte",
                        "Arrivées et services réguliers : aucune attente en régime établi.", None)
        if K == INF:
            return _rien("Système instable (λ > c·μ) : la file diverge.")
        pb = 1 - cap / lam
        return _fin(float(K), K - c, cap, ES, c, pb, 0.0, 0.0, "approchée",
                    "Système saturé : serveurs toujours occupés, la salle reste (presque) pleine.", None)
    lam_a = lam * (1 - q)
    if q > 0 and A != "M":
        return _rien("Refus aléatoires avec arrivées non poissonniennes : simulation seule.")
    if K == INF:
        if c == INF:
            return _fin(lam_a * ES, 0.0, lam_a, ES, c, 0.0, q, 0.0, "exacte",
                        "Infinité de serveurs : L = λ·E[S] quelle que soit la loi (formule de Little).")
        rho = lam_a / (c * mu)
        if rho >= 1:
            return _rien("Système instable (ρ ≥ 1) : la file grossit sans limite.")
        if c == 1 and A == "M":
            wq = lam_a * (1 + cs2) * ES * ES / (2 * (1 - rho))
            return _fin(lam_a * (wq + ES), lam_a * wq, lam_a, ES, 1, 0.0, q, 0.0, "exacte",
                        "Formule de Pollaczek-Khinchine : Wq = λ·E[S²] / (2(1−ρ)).")
        if c == 1 and B == "M" and A in ("D", "E", "H", "U"):
            sg = _gm1_sigma(p)
            wq = sg / (mu * (1 - sg))
            return _fin(lam_a * (wq + ES), lam_a * wq, lam_a, ES, 1, 0.0, q, 0.0, "exacte",
                        "G/M/1 : racine σ de l'équation σ = A*(μ(1−σ)) (transformée de Laplace des arrivées).")
        # Kingman / Allen-Cunneen
        mm = dict(p, arr="M", srv="M", balk=0.0, lam=lam_a, patience=None)
        base = _naissance_mort(mm)
        wq = base["Wq"] * (ca2 + cs2) / 2.0
        nom = "Kingman" if c == 1 else "Allen-Cunneen"
        return _fin(lam_a * (wq + ES), lam_a * wq, lam_a, ES, c, 0.0, q, 0.0, "approchée",
                    f"Approximation de {nom} : Wq ≈ Wq(M/M/c) · (ca² + cs²) / 2.")
    if c == 1 and A == "M" and B in ("M", "D", "E", "H"):
        r = _mg1k(p)
        if r:
            return r
    return _rien("Capacité finie avec lois générales : simulation seule.")


# ======================================================================
#  Thème, polices, couleurs
# ======================================================================
class T:
    BG = "#F7F2E7"
    CARD = "#FFFEFA"
    BORDER = "#D8C89F"
    SHADOW = "#E5D8B8"
    PRIMARY = "#C49A3A"
    PRIMARY_D = "#805B12"
    PRIMARY_L = "#E8D39A"
    TEXT = "#302719"
    MUTED = "#756A58"
    DISABLED = "#B8B8BE"
    RED = "#FF3B30"
    ORANGE = "#FF9500"
    GREEN = "#32ADE6"
    BLUE = "#5E5CE6"
    TIP = "#2C2C2E"


FAMILY = "Helvetica"
MONO = "Courier"


def init_theme(root):
    global FAMILY, MONO
    FAMILY = tkfont.nametofont("TkDefaultFont").actual("family")
    dispo = set(tkfont.families(root))
    for f in ("SF Pro Text", "SF Pro Display", ".SF NS Text", "Helvetica Neue", "Segoe UI Variable Text",
              "Segoe UI", "Inter", "Ubuntu", "Noto Sans", "DejaVu Sans"):
        if f in dispo:
            FAMILY = f
            break
    MONO = tkfont.nametofont("TkFixedFont").actual("family")


def fnt(size=10, weight="normal"):
    return (FAMILY, size, weight)


_polices = {}


def mesure(texte, police):
    f = _polices.get(police)
    if f is None:
        f = _polices[police] = tkfont.Font(font=police)
    return f.measure(texte)


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def mix(c1, c2, t):
    a, b = hex_rgb(c1), hex_rgb(c2)
    return "#%02x%02x%02x" % tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def texte_sur(col):
    r, g, b = hex_rgb(col)
    return "#FFFFFF" if 0.299 * r + 0.587 * g + 0.114 * b < 150 else "#1F2937"


def rrect(c, x0, y0, x1, y1, r, **kw):
    """Rectangle arrondi (polygone lissé)."""
    r = max(0, min(r, (x1 - x0) / 2, (y1 - y0) / 2))
    p = [x0 + r, y0, x0 + r, y0, x1 - r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y0 + r,
         x1, y1 - r, x1, y1 - r, x1, y1, x1 - r, y1, x1 - r, y1, x0 + r, y1, x0 + r, y1,
         x0, y1, x0, y1 - r, x0, y1 - r, x0, y0 + r, x0, y0 + r, x0, y0]
    return c.create_polygon(p, smooth=True, **kw)


def dims(c):
    return max(c.winfo_width(), 160), max(c.winfo_height(), 110)


# ======================================================================
#  Icônes dessinées (vectorielles, sur Canvas) — grille de 24 unités
# ======================================================================
def _icone_base(c, name, cx, cy, size, col, tag="ico"):
    u = size / 24.0

    def X(x):
        return cx + x * u

    def Y(y):
        return cy + y * u

    def pts(p):
        return [v for i in range(0, len(p), 2) for v in (X(p[i]), Y(p[i + 1]))]

    def line(*p, w=2.0):
        c.create_line(*pts(p), fill=col, width=max(1, w * u), capstyle="round",
                      joinstyle="round", tags=tag)

    def poly(*p, fill=True, w=2.0):
        c.create_polygon(*pts(p), fill=col if fill else "", outline=col,
                         width=max(1, w * u), joinstyle="round", tags=tag)

    def circ(x, y, r, fill=False, w=2.0):
        c.create_oval(X(x - r), Y(y - r), X(x + r), Y(y + r),
                      fill=col if fill else "", outline="" if fill else col,
                      width=max(1, w * u), tags=tag)

    def arc(x, y, r, start, extent, w=2.0):
        c.create_arc(X(x - r), Y(y - r), X(x + r), Y(y + r), start=start, extent=extent,
                     style="arc", outline=col, width=max(1, w * u), tags=tag)

    n = name
    if n == "play":
        poly(-4, -8, -4, 8, 9, 0, w=3)
    elif n == "pause":
        line(-4.5, -6, -4.5, 6, w=5)
        line(4.5, -6, 4.5, 6, w=5)
    elif n == "step":
        poly(-8, -7, -8, 7, 3, 0, w=3)
        line(7.5, -7, 7.5, 7, w=3)
    elif n == "stop":
        rrect(c, X(-6.5), Y(-6.5), X(6.5), Y(6.5), 2.5 * u, fill=col, outline="", tags=tag)
    elif n == "shuffle":
        line(-9, -5, -4, -5, 4, 5, 6, 5)
        line(-9, 5, -4, 5, 4, -5, 6, -5)
        poly(6, -9, 10.5, -5, 6, -1, w=1.5)
        poly(6, 1, 10.5, 5, 6, 9, w=1.5)
    elif n == "reset":
        arc(0, 0, 7, 90, 270, w=2.2)
        poly(3.5, -0.5, 10.5, -0.5, 7, -5, w=1.5)
    elif n == "download":
        line(0, -9, 0, 2)
        poly(-5, -1.5, 5, -1.5, 0, 4.5, w=1.5)
        line(-8, 3, -8, 8, 8, 8, 8, 3)
    elif n == "save":
        rrect(c, X(-8), Y(-8), X(8), Y(8), 2.5 * u, fill="", outline=col,
              width=max(1, 2 * u), tags=tag)
        line(-4, -8, -4, -3, 4, -3, 4, -8)
        line(-4, 8, -4, 2, 4, 2, 4, 8)
    elif n == "trash":
        line(-8, -5, 8, -5)
        line(-3, -5, -3, -8, 3, -8, 3, -5)
        poly(-6, -3, 6, -3, 5, 8, -5, 8, fill=False)
        line(-2, 0, -2, 5)
        line(2, 0, 2, 5)
    elif n == "help":
        circ(0, 0, 9.5)
        arc(0, -3.5, 3.3, 170, -250)
        line(0, -0.2, 0, 1.8)
        circ(0, 5.6, 1.1, fill=True)
    elif n == "more":
        circ(0, -6, 1.8, fill=True)
        circ(0, 0, 1.8, fill=True)
        circ(0, 6, 1.8, fill=True)
    elif n == "info":
        circ(0, 0, 9.5)
        circ(0, -4.5, 1.1, fill=True)
        line(0, -1, 0, 5.5)
    elif n == "stats":
        line(-6, 8, -6, 1, w=4)
        line(0, 8, 0, -7, w=4)
        line(6, 8, 6, -3, w=4)
    elif n == "compare":
        line(-8, -6, 3, -6, w=3.5)
        line(-8, 0, 8, 0, w=3.5)
        line(-8, 6, -1, 6, w=3.5)
    elif n == "sort":
        circ(-7.5, 4, 3, fill=True)
        circ(-1, 2.5, 4.5, fill=True)
        circ(6.5, 1.5, 5.5, fill=True)
    elif n == "book":
        poly(-10, -6, -1, -4, -1, 8, -10, 6, fill=False)
        poly(10, -6, 1, -4, 1, 8, 10, 6, fill=False)
    elif n == "bolt":
        poly(2, -9, -6, 2, 0, 2, -2, 9, 6, -2, 0, -2, w=1.5)
    elif n == "list":
        for y in (-6, 0, 6):
            circ(-8, y, 1.2, fill=True)
            line(-4, y, 9, y)
    elif n == "check":
        line(-7, 0, -2, 5, 7, -5, w=2.6)
    elif n == "close":
        line(-6, -6, 6, 6)
        line(6, -6, -6, 6)
    elif n == "swap":
        line(-8, -4, 5, -4)
        poly(4, -8, 9, -4, 4, 0, w=1.5)
        line(8, 4, -5, 4)
        poly(-4, 0, -9, 4, -4, 8, w=1.5)
    elif n == "pencil":
        poly(-8, 8, -7, 3.5, 4, -7.5, 8, -3.5, -3.5, 7, fill=False)
        line(1, -4.5, 5, -0.5)
    elif n == "move":
        line(-9, 0, 9, 0)
        line(0, -9, 0, 9)
        line(6, -3, 9, 0, 6, 3)
        line(-6, -3, -9, 0, -6, 3)
        line(-3, -6, 0, -9, 3, -6)
        line(-3, 6, 0, 9, 3, 6)
    elif n == "ruler":
        line(-9, 0, 9, 0)
        line(-9, -4, -9, 4)
        line(9, -4, 9, 4)
        line(-4.5, -2, -4.5, 2)
        line(0, -2.5, 0, 2.5)
        line(4.5, -2, 4.5, 2)
    elif n == "target":
        circ(0, 0, 9.5)
        circ(0, 0, 5)
        circ(0, 0, 1.5, fill=True)
    elif n == "clock":
        circ(0, 0, 9.5)
        line(0, -5, 0, 0, 4, 3)
    elif n == "percent":
        circ(-5, -5, 2.5)
        circ(5, 5, 2.5)
        line(6, -7, -6, 7)
    elif n == "layers":
        poly(0, -8, 9, -3, 0, 2, -9, -3, fill=False)
        line(-9, 2, 0, 7, 9, 2)
    elif n == "ban":
        circ(0, 0, 9)
        line(-6.4, -6.4, 6.4, 6.4)
    elif n == "keyboard":
        rrect(c, X(-10), Y(-6.5), X(10), Y(6.5), 2.5 * u, fill="", outline=col,
              width=max(1, 2 * u), tags=tag)
        for x in (-5.5, -1.8, 1.8, 5.5):
            circ(x, -2, 0.9, fill=True)
        line(-5, 2.5, 5, 2.5)


# ======================================================================
#  Info-bulles (hints)
# ======================================================================
class _Tip:
    def __init__(self):
        self.win = None
        self.job = None
        self.job_w = None
        self.text = None

    def attach(self, w, text, delay=450):
        """text : chaîne, ou fonction(event) -> chaîne | None."""
        w.bind("<Enter>", lambda e: self._planifier(w, e, text, delay), add="+")
        w.bind("<Motion>", lambda e: self._bouge(w, e, text, delay), add="+")
        w.bind("<Leave>", lambda e: self.cacher(), add="+")
        w.bind("<ButtonPress>", lambda e: self.cacher(), add="+")

    @staticmethod
    def _txt(text, e):
        try:
            return text(e) if callable(text) else text
        except Exception:
            return None

    def _planifier(self, w, e, text, delay):
        self._annuler()
        try:
            self.job_w = w
            self.job = w.after(delay, lambda: self._afficher(w, e, text))
        except tk.TclError:
            pass

    def _afficher(self, w, e, text):
        self.job = None
        t = self._txt(text, e)
        if t:
            self._montrer(w, e.x_root, e.y_root, t)

    def _bouge(self, w, e, text, delay):
        if self.win is None:
            self._planifier(w, e, text, delay)
            return
        t = self._txt(text, e)
        if not t:
            self.cacher()
        elif t != self.text:
            self.cacher()
            self._montrer(w, e.x_root, e.y_root, t)
        else:
            self._placer(e.x_root, e.y_root)

    def _montrer(self, w, x, y, t):
        self.text = t
        self.win = tk.Toplevel(w)
        self.win.wm_overrideredirect(True)
        try:
            self.win.attributes("-topmost", True)
        except tk.TclError:
            pass
        cadre = tk.Frame(self.win, bg=T.TIP)
        cadre.pack()
        tk.Label(cadre, text=t, bg=T.TIP, fg="#FFFFFF", font=fnt(9), justify="left",
                 wraplength=320, padx=12, pady=7).pack()
        self._placer(x, y)

    def _placer(self, x, y):
        if self.win is None:
            return
        self.win.update_idletasks()
        w, h = self.win.winfo_reqwidth(), self.win.winfo_reqheight()
        sw, sh = self.win.winfo_screenwidth(), self.win.winfo_screenheight()
        px = min(x + 14, sw - w - 8)
        py = y + 22
        if py + h > sh - 8:
            py = y - h - 12
        self.win.wm_geometry(f"+{max(0, px)}+{max(0, py)}")

    def _annuler(self):
        if self.job is not None:
            try:
                self.job_w.after_cancel(self.job)
            except Exception:
                pass
            self.job = None

    def cacher(self):
        self._annuler()
        if self.win is not None:
            try:
                self.win.destroy()
            except tk.TclError:
                pass
            self.win = None
            self.text = None


TIP = _Tip()


def tip_all(w, text):
    """Attache la même info-bulle à un widget et à tous ses descendants."""
    TIP.attach(w, text)
    for ch in w.winfo_children():
        tip_all(ch, text)


# ======================================================================
#  Widgets dessinés à la main (style application mobile)
# ======================================================================
KINDS = {
    "tonal": dict(bg=T.PRIMARY_L, hov="#D9BC72", prs=T.PRIMARY, fg="#352817",
                  off_bg="#E3E3E8", off_fg=T.DISABLED),
    "filled": dict(bg=T.PRIMARY_L, hov="#D9BC72", prs=T.PRIMARY, fg="#352817",
                   off_bg="#C7C7CC", off_fg="#F4F6FA"),
    "danger": dict(bg="#FEE2E2", hov="#FECACA", prs="#FCA5A5", fg=T.RED,
                   off_bg="#E3E3E8", off_fg=T.DISABLED),
    "ghost": dict(bg=None, hov="#DDDDE3", prs="#CFCFD6", fg=T.MUTED,
                  off_bg=None, off_fg=T.DISABLED),
    "light": dict(bg=None, hov="#ECECF0", prs="#DCDCE2",
                  fg=T.PRIMARY, off_bg=None, off_fg=T.DISABLED),
}


class _Clickable(tk.Canvas):
    """Base : survol, appui, activation/désactivation, info-bulle."""

    def __init__(self, parent, command, tip, **kw):
        super().__init__(parent, highlightthickness=0, bd=0, bg=parent.cget("bg"), **kw)
        self.command = command
        self.tip = tip
        self.enabled = True
        self.hover = False
        self.pressed = False
        self.bind("<Enter>", self._enter, add="+")
        self.bind("<Leave>", self._leave, add="+")
        self.bind("<ButtonPress-1>", self._press, add="+")
        self.bind("<ButtonRelease-1>", self._release, add="+")
        TIP.attach(self, lambda e: self.tip)

    def _enter(self, _):
        self.hover = True
        self.configure(cursor="hand2" if self.enabled else "arrow")
        self._draw()

    def _leave(self, _):
        self.hover = False
        self.pressed = False
        self._draw()

    def _press(self, _):
        if self.enabled:
            self.pressed = True
            self._draw()

    def _release(self, e):
        was = self.pressed
        self.pressed = False
        self._draw()
        if (was and self.enabled and self.command
                and 0 <= e.x <= self.winfo_width() and 0 <= e.y <= self.winfo_height()):
            self.command()

    def set_enabled(self, on):
        self.enabled = bool(on)
        self.hover = False
        self.pressed = False
        self.configure(cursor="hand2" if self.enabled else "arrow")
        self._draw()

    def _couleurs(self):
        k = KINDS[self.kind]
        if not self.enabled:
            return k["off_bg"], k["off_fg"]
        if self.pressed:
            return k["prs"], k["fg"]
        if self.hover:
            return k["hov"], k["fg"]
        return k["bg"], k["fg"]


class IconButton(_Clickable):
    """Bouton rond avec icône dessinée et libellé optionnel dessous."""

    def __init__(self, parent, icon, command, tip="", size=46, kind="tonal", label="",
                 icon_size=None, fab=False):
        lw = mesure(label, fnt(8)) + 14 if label else 0
        extra = 8 if fab else 0
        wd = max(size + extra, lw)
        ht = size + extra + (18 if label else 0)
        super().__init__(parent, command, tip, width=wd, height=ht)
        self.icon, self.sz, self.kind, self.label, self.fab = icon, size, kind, label, fab
        self._wd = wd
        self.isz = icon_size or int(size * 0.48)
        self._draw()

    def reconfigurer(self, icon=None, label=None, tip=None):
        if icon is not None:
            self.icon = icon
        if label is not None:
            self.label = label
        if tip is not None:
            self.tip = tip
        self._draw()

    def _draw(self):
        self.delete("all")
        fill, fg = self._couleurs()
        cx = self._wd / 2
        r = self.sz / 2
        cy = r + (2 if self.fab else 0)
        if self.fab and self.enabled:
            for k, col in ((5, mix(T.PRIMARY, "#FFFFFF", .82)), (3, mix(T.PRIMARY, "#FFFFFF", .68))):
                self.create_oval(cx - r + 1, cy - r + k, cx + r - 1, cy + r + k - 1, fill=col, outline="")
        if fill:
            self.create_oval(cx - r + 1, cy - r + 1, cx + r - 1, cy + r - 1, fill=fill, outline="")
        draw_icon(self, self.icon, cx, cy, self.isz, fg)
        if self.label:
            self.create_text(cx, cy + r + 10, text=self.label, font=fnt(8),
                             fill=T.MUTED if self.enabled else T.DISABLED)


class PillButton(_Clickable):
    """Bouton en pilule avec icône + texte."""

    def __init__(self, parent, text, command, tip="", icon=None, kind="filled", height=44):
        self.f = fnt(10, "bold")
        padx = 18
        wd = padx * 2 + mesure(text, self.f) + (30 if icon else 0)
        super().__init__(parent, command, tip, width=int(wd), height=height)
        self.text, self.icon, self.kind, self.ht, self._wd = text, icon, kind, height, int(wd)
        self._draw()

    def _draw(self):
        self.delete("all")
        fill, fg = self._couleurs()
        rrect(self, 1, 1, self._wd - 1, self.ht - 1, self.ht / 2, fill=fill or "", outline="")
        x = 18
        if self.icon:
            draw_icon(self, self.icon, x + 10, self.ht / 2, 20, fg)
            x += 30
        self.create_text(x, self.ht / 2, text=self.text, font=self.f, fill=fg, anchor="w")


class Toggle(tk.Canvas):
    """Interrupteur façon iOS lié à une BooleanVar."""

    def __init__(self, parent, variable, tip=""):
        super().__init__(parent, width=51, height=31, highlightthickness=0, bd=0,
                         bg=parent.cget("bg"), cursor="hand2")
        self.var = variable
        self.bind("<Button-1>", lambda e: self.var.set(not self.var.get()))
        self.var.trace_add("write", lambda *a: self._draw())
        TIP.attach(self, tip)
        self._draw()

    def _draw(self):
        self.delete("all")
        on = self.var.get()
        rrect(self, 1, 1, 50, 30, 14.5, fill=T.PRIMARY if on else "#D1D1D6", outline="")
        x = 36 if on else 16
        self.create_oval(x - 13, 4, x + 13, 30, fill="#C7C7CC", outline="")
        self.create_oval(x - 13, 2.5, x + 13, 28.5, fill="#FFFFFF", outline="#E3E3E8")


class ChipGroup(tk.Canvas):
    """Puces de sélection (segmented control), retour à la ligne automatique."""
    H = 34

    def __init__(self, parent, items, value, command, tips=None):
        police = fnt(9, "bold")
        wids = [mesure(lib, police) + 36 for _, lib in items]
        total = int(sum(wids) + 8 * (len(items) - 1))
        super().__init__(parent, width=total, height=self.H, bg=parent.cget("bg"),
                         highlightthickness=0, bd=0)
        self.items, self.value, self.command = items, value, command
        self.tips = tips or {}
        self.f = police
        self.wids = wids
        self.enabled = True
        self.hover = None
        self.rects = []
        self.lw = None
        self.base = T.BG if parent.cget("bg") == T.CARD else T.CARD
        self.bind("<Configure>", self._conf)
        self.bind("<Motion>", self._mouvement, add="+")
        self.bind("<Leave>", lambda e: self._survol(None), add="+")
        self.bind("<Button-1>", self._clic)
        TIP.attach(self, self._tip)

    def _conf(self, e):
        if e.width != self.lw:
            self.lw = e.width
            self._disposer(e.width)
        self._draw()

    def _disposer(self, largeur):
        self.rects = []
        x = y = 0
        for w in self.wids:
            if x + w > largeur and x > 0:
                x, y = 0, y + self.H + 8
            self.rects.append((x, y, x + w, y + self.H))
            x += w + 8
        haut = y + self.H
        if int(float(self.cget("height"))) != haut:
            self.configure(height=haut)

    def _cle_en(self, x, y):
        for (cle, _), (x0, y0, x1, y1) in zip(self.items, self.rects):
            if x0 <= x <= x1 and y0 <= y <= y1:
                return cle
        return None

    def _mouvement(self, e):
        self._survol(self._cle_en(e.x, e.y))

    def _survol(self, cle):
        if cle != self.hover:
            self.hover = cle
            self.configure(cursor="hand2" if cle and self.enabled else "arrow")
            self._draw()

    def _tip(self, e):
        return self.tips.get(self._cle_en(e.x, e.y))

    def _clic(self, e):
        cle = self._cle_en(e.x, e.y)
        if cle and self.enabled:
            self.set(cle, notifier=True)

    def set(self, cle, notifier=False):
        self.value = cle
        self._draw()
        if notifier and self.command:
            self.command(cle)

    def set_enabled(self, on):
        self.enabled = bool(on)
        self._draw()

    def _draw(self):
        self.delete("all")
        if not self.rects:
            return
        for (cle, lib), (x0, y0, x1, y1) in zip(self.items, self.rects):
            sel = cle == self.value
            if sel:
                fill = T.PRIMARY if self.enabled else mix(T.PRIMARY, "#FFFFFF", .5)
                outline, fg = "", "#FFFFFF"
            else:
                fill = mix(self.base, T.PRIMARY, .08) if (cle == self.hover and self.enabled) else self.base
                outline = T.BORDER
                fg = T.TEXT if self.enabled else T.DISABLED
            rrect(self, x0 + 1, y0 + 1, x1 - 1, y1 - 1, self.H / 2, fill=fill, outline=outline)
            cx = (x0 + x1) / 2
            self.create_text(cx, (y0 + y1) / 2, text=lib, font=self.f, fill=fg)


class Card(tk.Canvas):
    """Carte arrondie avec ombre douce ; le contenu va dans `.body`."""

    def __init__(self, parent, pad=14, radius=22, expand=False):
        super().__init__(parent, bg=parent.cget("bg"), highlightthickness=0, bd=0, height=10)
        self.pad, self.radius, self.expand = pad, radius, expand
        self.body = tk.Frame(self, bg=T.CARD)
        self.win = self.create_window(pad, pad, anchor="nw", window=self.body)
        self.bind("<Configure>", self._conf)
        self.body.bind("<Configure>", lambda e: self._sync())
        self.after_idle(self._sync)

    def _sync(self):
        if not self.expand:
            self.configure(height=self.body.winfo_reqheight() + 2 * self.pad + 4)

    def _conf(self, e):
        w, h = e.width, e.height
        self.delete("fond")
        rrect(self, 2, 3, w - 1, h - 1, self.radius, fill=T.SHADOW, outline="", tags="fond")
        rrect(self, 1, 1, w - 3, h - 4, self.radius, fill=T.CARD, outline="", tags="fond")
        self.tag_lower("fond")
        self.itemconfigure(self.win, width=max(1, w - 2 * self.pad - 4),
                           height=max(1, h - 2 * self.pad - 4))
        self.coords(self.win, self.pad + 1, self.pad + 1)


class NavItem(tk.Canvas):
    """Destination de la barre de navigation du bas."""

    def __init__(self, parent, icon, label, command, tip):
        super().__init__(parent, height=66, bg=T.CARD, highlightthickness=0, bd=0, cursor="hand2")
        self.icon, self.label, self.command = icon, label, command
        self.active = False
        self.hover = False
        self.bind("<Configure>", lambda e: self._draw())
        self.bind("<Enter>", lambda e: self._survol(True))
        self.bind("<Leave>", lambda e: self._survol(False))
        self.bind("<Button-1>", lambda e: self.command())
        TIP.attach(self, tip)

    def _survol(self, v):
        self.hover = v
        self._draw()

    def set_active(self, a):
        self.active = a
        self._draw()

    def _draw(self):
        self.delete("all")
        w = max(self.winfo_width(), 80)
        cx = w / 2
        col = T.PRIMARY if self.active else (T.TEXT if self.hover else T.MUTED)
        draw_icon(self, self.icon, cx, 24, 25, col)
        self.create_text(cx, 51, text=self.label, font=fnt(9, "bold" if self.active else "normal"), fill=col)


# ======================================================================
#  Graphiques dessinés sur Canvas (info-bulle au survol)
# ======================================================================
def fmt(v):
    return f"{v:,.0f}".replace(",", " ")


def echelle(mx):
    if mx <= 4:
        return 4
    mag = 10 ** math.floor(math.log10(mx / 4))
    for m in (1, 2, 5, 10):
        if m * mag * 4 >= mx:
            return m * mag * 4
    return mx


def _grille(c, w, h, ml, mr, mt, mb, ech):
    pw, ph = w - ml - mr, h - mt - mb
    for k in range(5):
        y = mt + ph - ph * k / 4
        c.create_line(ml, y, w - mr, y, fill=T.BORDER, dash=(3, 4) if k else None)
        c.create_text(ml - 7, y, text=fmt(ech * k / 4), anchor="e", font=fnt(7), fill=T.MUTED)
    return pw, ph


def _barre(c, x0, y0, x1, y1, col):
    """Barre à sommet arrondi."""
    if x1 - x0 >= 5 and y1 - y0 >= 4:
        r = min(4, (x1 - x0) / 2)
        rrect(c, x0, y0, x1, y1, r, fill=col, outline="")
        c.create_rectangle(x0, max(y0 + r, y1 - r), x1, y1, fill=col, outline="")
    else:
        c.create_rectangle(x0, y0, max(x1, x0 + 1), y1, fill=col, outline="")


def _legende(c, series, x=12, y=14):
    for libelle, col in series:
        c.create_oval(x, y - 5, x + 10, y + 5, fill=col, outline="")
        t = c.create_text(x + 15, y, text=libelle, anchor="w", font=fnt(8), fill=T.TEXT)
        x = c.bbox(t)[2] + 14


def hit_text(c, e):
    f = getattr(c, "hitf", None)
    return f(e) if f else None


def graphique_barres(c, data, couleur, unite):
    """Histogramme : une barre par case ; la plus haute est mise en évidence."""
    c.delete("all")
    c.hitf = None
    w, h = dims(c)
    ml, mr, mt, mb = 40, 10, 12, 24
    n = len(data)
    mx = max(data) if data else 0
    ech = echelle(mx)
    pw, ph = _grille(c, w, h, ml, mr, mt, mb, ech)
    if n == 0:
        return
    bw = pw / n
    marge = max(0.5, bw * 0.12)
    for i, v in enumerate(data):
        if v > 0:
            _barre(c, ml + i * bw + marge, mt + ph - ph * v / ech,
                   ml + (i + 1) * bw - marge, mt + ph, T.RED if v == mx else couleur)
    c.create_line(ml, mt + ph, w - mr, mt + ph, fill="#94A3B8")
    pas = max(1, n // 10)
    for i in range(0, n, pas):
        c.create_text(ml + (i + 0.5) * bw, mt + ph + 11, text=str(i + 1), font=fnt(7), fill=T.MUTED)

    def hit(e):
        if ml <= e.x <= ml + pw and mt <= e.y <= mt + ph:
            i = min(n - 1, int((e.x - ml) / bw))
            return f"Case {i + 1} : {data[i]} {unite}"
    c.hitf = hit


def graphique_courbes(c, series, xlabel="étapes"):
    """Courbes cumulées avec aire teintée. series = [(libellé, données, couleur)]"""
    c.delete("all")
    c.hitf = None
    w, h = dims(c)
    ml, mr, mt, mb = 46, 12, 34, 24
    longueur = max((len(d) for _, d, _ in series), default=0)
    mx = max((max(d) for _, d, _ in series if d), default=0)
    ech = echelle(mx)
    xmax = max(longueur - 1, 1)
    pw, ph = _grille(c, w, h, ml, mr, mt, mb, ech)
    for k in range(3):
        c.create_text(ml + pw * k / 2, mt + ph + 11, text=fmt(xmax * k / 2), font=fnt(7), fill=T.MUTED)
    c.create_text(w - mr, mt + ph + 11, text=xlabel, anchor="e", font=fnt(7), fill="#94A3B8")
    c.create_line(ml, mt + ph, w - mr, mt + ph, fill="#94A3B8")
    for _, d, col in series:
        if len(d) < 2:
            continue
        step = max(1, len(d) // max(1, int(pw)))
        idx = list(range(0, len(d), step))
        if idx[-1] != len(d) - 1:
            idx.append(len(d) - 1)
        pts = []
        for i in idx:
            pts += [ml + pw * i / xmax, mt + ph - ph * d[i] / ech]
        if len(series) <= 2:
            c.create_polygon(ml, mt + ph, *pts, pts[-2], mt + ph, fill=mix(col, "#FFFFFF", .86), outline="")
        c.create_line(*pts, fill=col, width=2.5, capstyle="round", joinstyle="round")
        c.create_oval(pts[-2] - 4, pts[-1] - 4, pts[-2] + 4, pts[-1] + 4, fill="#FFFFFF", outline=col, width=2)
    _legende(c, [(lib, col) for lib, _, col in series])

    def hit(e):
        if longueur < 2 or not (ml <= e.x <= ml + pw):
            return None
        i = max(0, min(longueur - 1, int(round((e.x - ml) / pw * xmax))))
        lignes = [f"Étape {i}"]
        for lib, d, _ in series:
            if d:
                lignes.append(f"{lib} : {fmt(d[min(i, len(d) - 1)])}" + ("  (terminé)" if i >= len(d) else ""))
        return "\n".join(lignes)
    c.hitf = hit


def graphique_groupes(c, categories, series):
    """Barres groupées. series = [(libellé, valeurs, couleur)]"""
    c.delete("all")
    c.hitf = None
    w, h = dims(c)
    ml, mr, mt, mb = 52, 10, 34, 26
    if not categories or not series:
        return
    mx = max((max(v) for _, v, _ in series if v), default=0)
    ech = echelle(mx)
    pw, ph = _grille(c, w, h, ml, mr, mt, mb, ech)
    ng, ns = len(categories), len(series)
    gw = pw / ng
    bw = gw * 0.74 / ns
    for g in range(ng):
        for j, (_, vals, col) in enumerate(series):
            v = vals[g]
            x0 = ml + g * gw + gw * 0.13 + j * bw
            if v > 0:
                _barre(c, x0, mt + ph - ph * v / ech, x0 + bw - 2, mt + ph, col)
            if bw > 30:
                c.create_text(x0 + bw / 2, mt + ph - ph * v / ech - 8, text=fmt(v), font=fnt(7), fill=T.TEXT)
        c.create_text(ml + (g + 0.5) * gw, mt + ph + 12, text=categories[g], font=fnt(8), fill=T.MUTED)
    c.create_line(ml, mt + ph, w - mr, mt + ph, fill="#94A3B8")
    _legende(c, [(lib, col) for lib, _, col in series])

    def hit(e):
        if ml <= e.x <= ml + pw and mt <= e.y <= mt + ph:
            g = min(ng - 1, int((e.x - ml) / gw))
            return categories[g] + "\n" + "\n".join(f"{lib} : {fmt(v[g])}" for lib, v, _ in series)
    c.hitf = hit



# ======================================================================
#  Widgets supplémentaires, icônes, graphiques spécifiques aux files d'attente
# ======================================================================
def draw_icon(c, name, cx, cy, size, col, tag="ico"):
    """Icônes de l'application (les icônes génériques viennent de _icone_base)."""
    u = size / 24.0

    def X(x):
        return cx + x * u

    def Y(y):
        return cy + y * u

    def pts(p):
        return [v for i in range(0, len(p), 2) for v in (X(p[i]), Y(p[i + 1]))]

    def line(*p, w=2.0):
        c.create_line(*pts(p), fill=col, width=max(1, w * u), capstyle="round", joinstyle="round", tags=tag)

    def poly(*p, fill=True, w=2.0):
        c.create_polygon(*pts(p), fill=col if fill else "", outline=col, width=max(1, w * u),
                         joinstyle="round", tags=tag)

    def circ(x, y, r, fill=False, w=2.0):
        c.create_oval(X(x - r), Y(y - r), X(x + r), Y(y + r), fill=col if fill else "",
                      outline="" if fill else col, width=max(1, w * u), tags=tag)

    def arc(x, y, r, start, extent, w=2.0):
        c.create_arc(X(x - r), Y(y - r), X(x + r), Y(y + r), start=start, extent=extent, style="arc",
                     outline=col, width=max(1, w * u), tags=tag)

    n = name
    if n == "model":
        for y, x in ((-6, -4), (0, 4), (6, -1)):
            line(-9, y, 9, y, w=1.6)
            circ(x, y, 2.7, fill=True)
    elif n == "queue":
        circ(-8, 0, 2.5, fill=True)
        circ(-2.5, 0, 2.5, fill=True)
        poly(3, -6.5, 10, -6.5, 10, 6.5, 3, 6.5, fill=False, w=1.8)
    elif n == "flask":
        line(-3, -9, 3, -9)
        poly(-2, -9, -2, -2, -8, 7, 8, 7, 2, -2, 2, -9, fill=False, w=1.9)
        line(-5, 3, 5, 3, w=1.6)
    elif n == "star":
        pp = []
        for k in range(10):
            a = -math.pi / 2 + k * math.pi / 5
            r = 10 if k % 2 == 0 else 4.4
            pp += [r * math.cos(a), r * math.sin(a)]
        poly(*pp, w=1.2)
    elif n == "users":
        circ(0, -4, 3.6, fill=True)
        arc(0, 11, 8, 25, 130, w=3)
    elif n == "gauge":
        arc(0, 4, 9, 0, 180)
        line(0, 4, 5, -3)
        circ(0, 4, 1.5, fill=True)
    elif n == "dice":
        rrect(c, X(-8.5), Y(-8.5), X(8.5), Y(8.5), 3 * u, fill="", outline=col, width=max(1, 2 * u), tags=tag)
        for x, y in ((-3.5, -3.5), (3.5, 3.5), (0, 0), (3.5, -3.5), (-3.5, 3.5)):
            circ(x, y, 1.1, fill=True)
    elif n == "ff":
        poly(-10, -6, -10, 6, -1, 0, w=2)
        poly(0, -6, 0, 6, 9, 0, w=2)
    elif n == "chart":
        line(-9, -9, -9, 9, 9, 9)
        line(-6, 4, -1, -1, 3, 2, 8, -6)
    elif n == "hourglass":
        line(-6, -9, 6, -9)
        line(-6, 9, 6, 9)
        poly(-5, -9, 5, -9, 0, 0, fill=False, w=1.8)
        poly(-5, 9, 5, 9, 0, 0, fill=False, w=1.8)
    elif n == "door":
        poly(-6, -9, 6, -9, 6, 9, -6, 9, fill=False, w=1.9)
        circ(3, 0, 1, fill=True)
    else:
        _icone_base(c, name, cx, cy, size, col, tag)


def draw_logo(c, size):
    """Logo : trois clients qui font la queue devant un guichet."""
    c.delete("all")
    u = size / 44.0
    rrect(c, 1, 1, size - 1, size - 1, 12 * u, fill=T.PRIMARY_D, outline=mix(T.PRIMARY_D, "#FFFFFF", .25))
    for x, col in ((9, "#BFE0FF"), (18, "#FCD34D"), (27, "#FDA4AF")):
        r = 4.2 * u
        c.create_oval((x - 4.2) * u, 26 * u - r, (x + 4.2) * u, 26 * u + r, fill=col, outline="")
    rrect(c, 31 * u, 10 * u, 40 * u, 34 * u, 3 * u, fill="#FFFFFF", outline="")
    c.create_oval(33.5 * u, 15 * u, 37.5 * u, 19 * u, fill=T.PRIMARY, outline="")
    c.create_line(5 * u, 36 * u, 40 * u, 36 * u, fill=mix(T.PRIMARY_D, "#FFFFFF", .5), width=max(1, 2 * u))


class Slider(tk.Canvas):
    """Curseur (entier ou décimal) à la façon mobile : piste arrondie, pouce rond avec halo."""
    PAD = 14

    def __init__(self, parent, lo, hi, value, command=None, tip="", step=1):
        super().__init__(parent, height=30, width=140, highlightthickness=0, bd=0, bg=parent.cget("bg"))
        self.lo, self.hi, self.step, self.command = lo, hi, step, command
        self.enabled = True
        self.drag = False
        self.value = self._snap(value)
        self.bind("<Configure>", lambda e: self._draw())
        self.bind("<ButtonPress-1>", self._down)
        self.bind("<B1-Motion>", self._move)
        self.bind("<ButtonRelease-1>", self._up)
        TIP.attach(self, tip)

    def _snap(self, v):
        v = min(self.hi, max(self.lo, v))
        v = round(self.lo + round((v - self.lo) / self.step) * self.step, 6)
        if all(isinstance(x, int) for x in (self.lo, self.step)):
            return int(v)
        return float(v)

    def get(self):
        return self.value

    def set(self, v):
        self.value = self._snap(v)
        self._draw()

    def regler(self, lo, hi, step, value):
        self.lo, self.hi, self.step = lo, hi, step
        self.value = self._snap(value)
        self._draw()

    def set_enabled(self, on):
        self.enabled = bool(on)
        self.configure(cursor="hand2" if on else "arrow")
        self._draw()

    def _x(self, v):
        w = max(self.winfo_width(), 60)
        return self.PAD + (w - 2 * self.PAD) * (v - self.lo) / ((self.hi - self.lo) or 1)

    def _maj(self, e):
        w = max(self.winfo_width(), 60)
        t = max(0.0, min(1.0, (e.x - self.PAD) / (w - 2 * self.PAD)))
        v = self._snap(self.lo + t * (self.hi - self.lo))
        if v != self.value:
            self.value = v
            self._draw()
            if self.command:
                self.command(v)

    def _down(self, e):
        if self.enabled:
            self.drag = True
            self._maj(e)
            self._draw()

    def _move(self, e):
        if self.enabled and self.drag:
            self._maj(e)

    def _up(self, e):
        self.drag = False
        self._draw()

    def _draw(self):
        self.delete("all")
        w = max(self.winfo_width(), 60)
        cy = 15
        x = self._x(self.value)
        col = T.PRIMARY if self.enabled else T.DISABLED
        self.create_line(self.PAD, cy, w - self.PAD, cy, width=4, fill="#D1D1D6", capstyle="round")
        self.create_line(self.PAD, cy, x, cy, width=4, fill=col, capstyle="round")
        self.create_oval(x - 12, cy - 10, x + 12, cy + 14, fill="#CFCFD4", outline="")
        self.create_oval(x - 12, cy - 12, x + 12, cy + 12, fill="#FFFFFF", outline="#E3E3E8")


class ScrollFrame(tk.Frame):
    """Page à défilement vertical ; le contenu se place dans `.inner`."""

    def __init__(self, parent):
        super().__init__(parent, bg=T.BG)
        self.cv = tk.Canvas(self, bg=T.BG, highlightthickness=0, bd=0)
        self.sb = tk.Scrollbar(self, orient="vertical", command=self.cv.yview, width=10, relief="flat",
                               troughcolor=T.BG, bd=0)
        self.cv.configure(yscrollcommand=self.sb.set)
        self.sb.pack(side="right", fill="y")
        self.cv.pack(side="left", fill="both", expand=True)
        self.inner = tk.Frame(self.cv, bg=T.BG)
        self.win = self.cv.create_window(0, 0, anchor="nw", window=self.inner)
        self.inner.bind("<Configure>", lambda e: self.cv.configure(scrollregion=self.cv.bbox("all")))
        self.cv.bind("<Configure>", lambda e: self.cv.itemconfigure(self.win, width=e.width))

    def scroll(self, d):
        self.cv.yview_scroll(d, "units")


class Reglage:
    """Un réglage : titre, valeur affichée, curseur et (option) interrupteur « ∞ »."""

    def __init__(self, app, parent, titre, lo, hi, val, step, fmtf, cmd, tip, icon=None, inf=False):
        self.app, self.fmt, self.cmd = app, fmtf, cmd
        self.frame = tk.Frame(parent, bg=T.CARD)
        haut = tk.Frame(self.frame, bg=T.CARD)
        haut.pack(fill="x")
        if icon:
            ic = tk.Canvas(haut, width=22, height=22, bg=T.CARD, highlightthickness=0)
            ic.pack(side="left")
            draw_icon(ic, icon, 11, 11, 18, T.PRIMARY)
        self.titre = tk.Label(haut, text=titre, font=fnt(9, "bold"), fg=T.TEXT, bg=T.CARD)
        self.titre.pack(side="left", padx=(4, 0))
        self.label = tk.Label(haut, text=fmtf(val), font=fnt(9, "bold"), fg=T.PRIMARY, bg=T.CARD)
        self.label.pack(side="right")
        self.var = None
        if inf:
            self.var = tk.BooleanVar(value=False)
            lab = tk.Label(haut, text="∞", font=fnt(11, "bold"), fg=T.MUTED, bg=T.CARD)
            lab.pack(side="right", padx=(0, 6))
            tg = Toggle(haut, self.var, "Illimité (∞) : désactive la limite pour ce paramètre.")
            tg.pack(side="right", padx=(0, 2))
            self.var.trace_add("write", lambda *a: self._inf())
        self.slider = Slider(self.frame, lo, hi, val, self._chg, tip, step)
        self.slider.pack(fill="x")
        tip_all(haut, tip)

    def _chg(self, v):
        self.label.config(text=self.fmt(v))
        if not self.app._sync:
            self.cmd(v)

    def _inf(self):
        on = self.var.get()
        self.slider.set_enabled(not on)
        self.label.config(text="∞" if on else self.fmt(self.slider.get()))
        if not self.app._sync:
            self.cmd(INF if on else self.slider.get())

    def set(self, v):
        if v == INF and self.var is not None:
            self.var.set(True)
            self.slider.set_enabled(False)
            self.label.config(text="∞")
            return
        if self.var is not None:
            self.var.set(False)
            self.slider.set_enabled(True)
        self.slider.set(v)
        self.label.config(text=self.fmt(self.slider.get()))

    def enable(self, on):
        self.slider.set_enabled(on)


# ----------------------------------------------------------------------
#  Graphiques
# ----------------------------------------------------------------------
def _msg(c, texte, icone="chart"):
    w, h = dims(c)
    draw_icon(c, icone, w / 2, h / 2 - 16, 30, T.DISABLED)
    c.create_text(w / 2, h / 2 + 14, fill=T.MUTED, font=fnt(9), text=texte)


def _ech_nice(v):
    if v <= 0:
        return 1.0
    mag = 10 ** math.floor(math.log10(v))
    for m in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if m * mag >= v:
            return m * mag
    return 10 * mag


def _grille_dec(c, w, h, ml, mr, mt, mb, ech):
    pw, ph = w - ml - mr, h - mt - mb
    for k in range(5):
        y = mt + ph - ph * k / 4
        c.create_line(ml, y, w - mr, y, fill=T.BORDER, dash=(3, 4) if k else None)
        c.create_text(ml - 7, y, text=f"{ech * k / 4:.3g}", anchor="e", font=fnt(7), fill=T.MUTED)
    return pw, ph


def graphique_traj(c, sim, clock, fenetre):
    """Nombre de clients dans le système au cours du temps (fenêtre glissante)."""
    c.delete("all")
    c.hitf = None
    w, h = dims(c)
    ml, mr, mt, mb = 32, 8, 8, 18
    if sim is None or len(sim.traj_t) < 2:
        _msg(c, "La courbe n(t) apparaît dès le départ", "chart")
        return
    t1 = max(clock, 1e-9)
    t0 = max(0.0, t1 - fenetre)
    pw, ph = w - ml - mr, h - mt - mb
    tt, nn = sim.traj_t, sim.traj_n
    i0 = max(0, bisect_right(tt, t0) - 1)
    nmax = max(max(nn[i0:]), sim.n(), 3)
    if sim.K != INF:
        nmax = max(nmax, sim.K)
    ech = max(4, int(math.ceil(nmax * 1.1)))
    for k in range(3):
        y = mt + ph - ph * k / 2
        c.create_line(ml, y, w - mr, y, fill=T.BORDER, dash=(3, 4) if k else None)
        c.create_text(ml - 5, y, text=f"{ech * k / 2:.0f}", anchor="e", font=fnt(7), fill=T.MUTED)
    pts = []
    cols = max(2, int(pw))
    for k in range(cols + 1):
        t = t0 + (t1 - t0) * k / cols
        j = bisect_right(tt, t) - 1
        n = nn[j] if j >= 0 else 0
        if k == cols and t >= tt[-1]:
            n = sim.n()
        x = ml + pw * k / cols
        y = mt + ph - ph * min(n, ech) / ech
        if pts:
            pts += [x, pts[-1]]
        pts += [x, y]
    c.create_polygon(ml, mt + ph, *pts, ml + pw, mt + ph, fill=mix(T.PRIMARY, "#FFFFFF", .85), outline="")
    c.create_line(*pts, fill=T.PRIMARY, width=2, joinstyle="round")
    if sim.c != INF and sim.c <= ech:
        y = mt + ph - ph * sim.c / ech
        c.create_line(ml, y, w - mr, y, fill=T.ORANGE, dash=(5, 3), width=1.5)
        c.create_text(w - mr - 2, y - 7, text=f"c = {int(sim.c)}", anchor="e", font=fnt(7), fill=T.ORANGE)
    if sim.K != INF and sim.K <= ech:
        y = mt + ph - ph * sim.K / ech
        c.create_line(ml, y, w - mr, y, fill=T.RED, dash=(5, 3), width=1.5)
        c.create_text(w - mr - 2, y - 7, text=f"K = {int(sim.K)}", anchor="e", font=fnt(7), fill=T.RED)
    c.create_line(ml, mt + ph, w - mr, mt + ph, fill="#94A3B8")
    c.create_text(ml, mt + ph + 9, text=f"{t0:.0f}", anchor="w", font=fnt(7), fill=T.MUTED)
    c.create_text(w - mr, mt + ph + 9, text=f"t = {t1:.0f}", anchor="e", font=fnt(7), fill=T.MUTED)


def graphique_pn(c, ps, pt):
    """Distribution du nombre de clients : simulation (barres) et théorie (points)."""
    c.delete("all")
    c.hitf = None
    if not ps:
        _msg(c, "Lancez une simulation ou une analyse")
        return
    w, h = dims(c)
    ml, mr, mt, mb = 40, 10, 32, 24
    n = min(max(len(ps), len(pt) if pt else 0), 41)
    ps = (list(ps) + [0.0] * n)[:n]
    pt = (list(pt) + [0.0] * n)[:n] if pt else None
    mx = max(max(ps), max(pt) if pt else 0) * 100
    ech = _ech_nice(max(mx, 1e-6) * 1.05)
    pw, ph = _grille_dec(c, w, h, ml, mr, mt, mb, ech)
    bw = pw / n
    for i, v in enumerate(ps):
        if v > 0:
            _barre(c, ml + i * bw + bw * .12, mt + ph - ph * v * 100 / ech, ml + (i + 1) * bw - bw * .12,
                   mt + ph, T.PRIMARY)
    if pt:
        pts = []
        for i, v in enumerate(pt):
            x, y = ml + (i + .5) * bw, mt + ph - ph * v * 100 / ech
            pts += [x, y]
        if len(pts) >= 4:
            c.create_line(*pts, fill=T.ORANGE, width=2, smooth=False)
        for i in range(0, len(pts), 2):
            c.create_oval(pts[i] - 3, pts[i + 1] - 3, pts[i] + 3, pts[i + 1] + 3, fill="#FFFFFF",
                          outline=T.ORANGE, width=2)
    c.create_line(ml, mt + ph, w - mr, mt + ph, fill="#94A3B8")
    pas = max(1, n // 12)
    for i in range(0, n, pas):
        c.create_text(ml + (i + .5) * bw, mt + ph + 11, text=str(i), font=fnt(7), fill=T.MUTED)
    _legende(c, [("Simulation", T.PRIMARY)] + ([("Théorie", T.ORANGE)] if pt else []))
    c.create_text(w - mr, 14, text="P(n) en %", anchor="e", font=fnt(7), fill=T.MUTED)

    def hit(e):
        if ml <= e.x <= ml + pw and mt <= e.y <= mt + ph:
            i = min(n - 1, int((e.x - ml) / bw))
            t = f"n = {i} client(s)\nSimulation : {ps[i] * 100:.2f} %"
            if pt:
                t += f"\nThéorie : {pt[i] * 100:.2f} %"
            return t
    c.hitf = hit


def graphique_hist(c, waits, moyenne):
    """Histogramme des temps d'attente avant service."""
    c.delete("all")
    c.hitf = None
    if not waits:
        _msg(c, "Aucune attente enregistrée")
        return
    w, h = dims(c)
    ml, mr, mt, mb = 40, 10, 32, 24
    ws = sorted(waits)
    q = ws[min(len(ws) - 1, int(len(ws) * 0.98))]
    hi = max(q, 1e-9)
    nb = 24
    bins = [0] * nb
    zero = 0
    for x in ws:
        if x <= 1e-12:
            zero += 1
        if x <= hi:
            bins[min(nb - 1, int(x / hi * nb))] += 1
    tot = len(ws)
    ech = _ech_nice(max(bins) / tot * 100 * 1.05)
    pw, ph = _grille_dec(c, w, h, ml, mr, mt, mb, ech)
    bw = pw / nb
    for i, v in enumerate(bins):
        if v:
            _barre(c, ml + i * bw + 1, mt + ph - ph * v / tot * 100 / ech, ml + (i + 1) * bw - 1, mt + ph, T.BLUE)
    c.create_line(ml, mt + ph, w - mr, mt + ph, fill="#94A3B8")
    for k in range(5):
        c.create_text(ml + pw * k / 4, mt + ph + 11, text=f"{hi * k / 4:.3g}", font=fnt(7), fill=T.MUTED)
    if moyenne == moyenne and 0 <= moyenne <= hi:
        x = ml + pw * moyenne / hi
        c.create_line(x, mt, x, mt + ph, fill=T.RED, dash=(4, 3), width=1.5)
        c.create_text(x + 4, mt + 6, text=f"moy. {moyenne:.2f}", anchor="w", font=fnt(7), fill=T.RED)
    _legende(c, [("Part des clients (%)", T.BLUE)])
    c.create_text(w - mr, 14, text=f"{zero / tot * 100:.0f} % sans attente", anchor="e", font=fnt(7), fill=T.MUTED)

    def hit(e):
        if ml <= e.x <= ml + pw and mt <= e.y <= mt + ph:
            i = min(nb - 1, int((e.x - ml) / bw))
            return f"Attente de {hi * i / nb:.2f} à {hi * (i + 1) / nb:.2f}\n{bins[i] / tot * 100:.2f} % des clients"
    c.hitf = hit


def graphique_xy(c, xs, series, xlabel, ylabel):
    """Courbes y = f(x). series = [dict(label, ys, col, ic=None, dash=False, pts=True)]"""
    c.delete("all")
    c.hitf = None
    vals = []
    for s in series:
        for i, y in enumerate(s["ys"]):
            if y is not None and y == y:
                vals += [y + (s["ic"][i] if s.get("ic") else 0)]
    if not xs or not vals:
        _msg(c, "Choisissez un balayage puis lancez-le", "flask")
        return
    w, h = dims(c)
    ml, mr, mt, mb = 56, 14, 34, 38
    ech = _ech_nice(max(max(vals), 1e-9) * 1.05)
    pw, ph = _grille_dec(c, w, h, ml, mr, mt, mb, ech)
    x0, x1 = min(xs), max(xs)
    if x1 == x0:
        x1 = x0 + 1

    def px(x):
        return ml + pw * (x - x0) / (x1 - x0)

    def py(y):
        return mt + ph - ph * min(y, ech) / ech
    c.create_line(ml, mt + ph, w - mr, mt + ph, fill="#94A3B8")
    for k in range(6):
        xv = x0 + (x1 - x0) * k / 5
        c.create_text(px(xv), mt + ph + 11, text=f"{xv:.3g}", font=fnt(7), fill=T.MUTED)
    c.create_text(ml + pw / 2, h - 8, text=xlabel, font=fnt(8), fill=T.MUTED)
    c.create_text(10, mt + ph / 2, text=ylabel, font=fnt(8), fill=T.MUTED, angle=90)
    for s in series:
        run = []
        for x, y in zip(xs, s["ys"]):
            if y is None or y != y:
                if len(run) >= 4:
                    c.create_line(*run, fill=s["col"], width=2.5, dash=(6, 4) if s.get("dash") else None,
                                  capstyle="round", joinstyle="round")
                run = []
            else:
                run += [px(x), py(y)]
        if len(run) >= 4:
            c.create_line(*run, fill=s["col"], width=2.5, dash=(6, 4) if s.get("dash") else None,
                          capstyle="round", joinstyle="round")
        if s.get("pts", True):
            for i, (x, y) in enumerate(zip(xs, s["ys"])):
                if y is None or y != y:
                    continue
                if s.get("ic"):
                    ic = s["ic"][i]
                    c.create_line(px(x), py(y - ic), px(x), py(y + ic), fill=s["col"], width=1.5)
                c.create_oval(px(x) - 3.5, py(y) - 3.5, px(x) + 3.5, py(y) + 3.5, fill="#FFFFFF",
                              outline=s["col"], width=2)
    _legende(c, [(s["label"], s["col"]) for s in series])

    def hit(e):
        if not (ml - 6 <= e.x <= ml + pw + 6):
            return None
        i = min(range(len(xs)), key=lambda k: abs(px(xs[k]) - e.x))
        lignes = [f"{xlabel} = {xs[i]:.3g}"]
        for s in series:
            y = s["ys"][i]
            if y is not None and y == y:
                ic = f" ± {s['ic'][i]:.3g}" if s.get("ic") else ""
                lignes.append(f"{s['label']} : {y:.4g}{ic}")
        return "\n".join(lignes)
    c.hitf = hit


class CompareTable(tk.Canvas):
    """Tableau Théorie / Simulation dessiné à la main avec verdict par ligne."""
    COLS = [("Indicateur", .32, "w"), ("Théorie", .17, "e"), ("Simulation (± IC 95 %)", .27, "e"),
            ("Écart", .13, "e"), ("Verdict", .11, "e")]
    HEAD, ROW = 34, 36

    def __init__(self, parent, nrows):
        super().__init__(parent, bg=T.CARD, highlightthickness=0, bd=0, height=self.HEAD + nrows * self.ROW + 6)
        self.rows, self.hover = [], -1
        self.bind("<Configure>", lambda e: self.draw())
        self.bind("<Motion>", lambda e: self._h(self._idx(e.y)))
        self.bind("<Leave>", lambda e: self._h(-1))
        TIP.attach(self, self._tip, delay=300)

    def set_rows(self, rows):
        self.rows = rows
        self.draw()

    def _idx(self, y):
        i = int((y - self.HEAD - 4) // self.ROW)
        return i if 0 <= i < len(self.rows) else -1

    def _h(self, i):
        if i != self.hover:
            self.hover = i
            self.draw()

    def _tip(self, e):
        i = self._idx(e.y)
        return self.rows[i]["tip"] if i >= 0 else None

    def draw(self):
        self.delete("all")
        w = max(self.winfo_width(), 420)
        if not self.rows:
            _msg(self, "Aucun résultat : lancez une simulation ou une analyse", "compare")
            return
        xs, x = [], 0
        for _, f, _ in self.COLS:
            xs.append((x, x + w * f))
            x += w * f
        rrect(self, 0, 0, w, self.HEAD, 10, fill=T.BG, outline="")
        for (titre, _, al), (a, b) in zip(self.COLS, xs):
            if al == "w":
                self.create_text(a + 14, self.HEAD / 2, text=titre, anchor="w", font=fnt(8, "bold"), fill=T.MUTED)
            else:
                self.create_text(b - 12, self.HEAD / 2, text=titre, anchor="e", font=fnt(8, "bold"), fill=T.MUTED)
        for r, ligne in enumerate(self.rows):
            y0 = self.HEAD + 4 + r * self.ROW
            yc = y0 + self.ROW / 2
            if r == self.hover:
                rrect(self, 0, y0 + 1, w, y0 + self.ROW - 1, 10, fill="#F2F2F5", outline="")
            elif r:
                self.create_line(14, y0, w - 14, y0, fill=T.BORDER)
            draw_icon(self, ligne["icon"], 26, yc, 18, ligne["col"])
            self.create_text(44, yc, text=ligne["nom"], anchor="w", font=fnt(10, "bold"), fill=T.TEXT)
            self.create_text(xs[1][1] - 12, yc, text=ligne["th"], anchor="e", font=fnt(10), fill=T.TEXT)
            self.create_text(xs[2][1] - 12, yc, text=ligne["sim"], anchor="e", font=fnt(10, "bold"), fill=T.TEXT)
            self.create_text(xs[3][1] - 12, yc, text=ligne["ecart"], anchor="e", font=fnt(9), fill=T.MUTED)
            vx = xs[4][1] - 22
            v = ligne["verdict"]
            if v == "ok":
                self.create_oval(vx - 11, yc - 11, vx + 11, yc + 11, fill=mix(T.GREEN, "#FFFFFF", .85), outline="")
                draw_icon(self, "check", vx, yc, 14, T.PRIMARY_D)
            elif v == "approx":
                self.create_oval(vx - 11, yc - 11, vx + 11, yc + 11, fill=mix(T.ORANGE, "#FFFFFF", .8), outline="")
                self.create_text(vx, yc - 1, text="≈", font=fnt(12, "bold"), fill="#B45309")
            elif v == "ko":
                self.create_oval(vx - 11, yc - 11, vx + 11, yc + 11, fill=mix(T.RED, "#FFFFFF", .85), outline="")
                draw_icon(self, "close", vx, yc, 12, "#B91C1C")
            else:
                self.create_text(vx, yc, text="—", font=fnt(11), fill=T.DISABLED)


# ======================================================================
#  Catalogue de modèles
# ======================================================================
PRESETS = [
    ("M/M/1", "Le modèle de base : arrivées de Poisson, services exponentiels, un seul guichet, file infinie.", dict()),
    ("M/M/1/K", "Un guichet et une capacité totale K : quand le système est plein, les nouveaux clients sont refusés.",
     dict(lam=0.9, K=6)),
    ("M/M/c", "Plusieurs serveurs identiques devant une file commune (caisses, centre d'appels).", dict(lam=2.4, c=3)),
    ("M/M/c/K", "Plusieurs serveurs et une salle d'attente limitée : mélange d'attente et de pertes.",
     dict(lam=1.6, c=2, K=8)),
    ("Erlang-B (M/M/c/c)", "Aucune salle d'attente : un appel est perdu si tous les circuits sont occupés (téléphonie).",
     dict(lam=3.0, c=4, K=4)),
    ("M/M/∞", "Autant de serveurs que de clients : personne n'attend, L = λ/μ (libre-service).",
     dict(lam=3.0, c=INF)),
    ("M/D/1", "Service de durée constante (machine, péage) : l'attente est moitié moindre qu'en M/M/1.",
     dict(srv="D")),
    ("M/D/1/K", "Service constant avec capacité limitée.", dict(srv="D", lam=0.9, K=6)),
    ("D/M/1", "Arrivées parfaitement régulières (cadence) et service exponentiel.", dict(arr="D")),
    ("D/D/1/K", "Tout est régulier mais λ > μ : le système sature et refuse 1 − μ/λ des clients.",
     dict(arr="D", srv="D", lam=1.2, K=5)),
    ("M/E₂/1", "Service en deux étapes (Erlang-2) : plus régulier que l'exponentielle.",
     dict(srv="E", srv_forme=2.0)),
    ("E₂/M/1", "Arrivées plus régulières que Poisson (Erlang-2).", dict(arr="E", arr_forme=2.0)),
    ("M/H₂/1", "Service très irrégulier (cv² = 4) : quelques clients très longs bloquent les autres.",
     dict(srv="H", srv_forme=4.0, lam=0.7)),
    ("M/G/1", "Service général (log-normal, cv² = 2) : formule de Pollaczek-Khinchine.",
     dict(srv="G", srv_forme=2.0)),
    ("M/G/1/K", "Service général (H₂) avec capacité K : résolu par chaîne de Markov incluse.",
     dict(srv="H", srv_forme=3.0, lam=0.9, K=8)),
    ("G/G/1", "Arrivées et services quelconques : approximation de Kingman.",
     dict(arr="G", arr_forme=1.5, srv="G", srv_forme=0.5)),
    ("G/G/c", "Plusieurs serveurs, lois générales : approximation d'Allen-Cunneen.",
     dict(arr="G", arr_forme=1.5, srv="G", srv_forme=2.0, c=2, lam=1.6)),
    ("M/M/1//N (population finie)", "N machines qui tombent en panne puis sont réparées par un technicien.",
     dict(N=6, lam=0.2)),
    ("M/M/c + abandons", "Les clients impatients quittent la file après une patience exponentielle.",
     dict(lam=2.2, c=2, patience=4.0)),
    ("M/M/1 + refus", "Certains clients renoncent à entrer (balking) avec une probabilité fixe.",
     dict(lam=1.2, balk=0.3)),
    ("M/M/1 LIFO", "Dernier arrivé, premier servi : mêmes moyennes que FIFO, mais variance des attentes plus grande.",
     dict(disc="LIFO")),
    ("M/M/1 SIRO", "Service dans un ordre aléatoire.", dict(disc="SIRO")),
    ("M/G/1 SJF", "Le plus court d'abord : réduit l'attente moyenne (pas de formule ici, simulation seule).",
     dict(srv="G", srv_forme=2.0, disc="SJF", lam=0.85)),
]


def params_preset(over):
    p = dict(DEFAUT)
    p.update(over)
    return normaliser(p)


# ======================================================================
#  Défis
# ======================================================================
DEFIS = [
    dict(id="caisse", titre="Les caisses du supermarché", icon="users", metrique="W", op="<=", niveaux=(3.0, 1.6, 1.2),
         unite="u.t. (temps total moyen)", base=dict(lam=3.0, mu=1.0, c=1),
         histoire="Le samedi, 3 clients par minute arrivent et chaque caissière en traite 1 par minute. "
                  "Le gérant dispose de 6 caisses au plus.",
         objectif="Temps moyen dans le système W ≤ 3 (★), ≤ 1,6 (★★), ≤ 1,2 (★★★) avec c ≤ 6.",
         contrainte=lambda p: (p["c"] != INF and p["c"] <= 6, "c ≤ 6 caisses"),
         indice="Il faut c·μ > λ. Chaque caisse en plus a un rendement décroissant."),
    dict(id="urgences", titre="Salle d'urgences", icon="gauge", metrique="p_block", op="<=", niveaux=(0.15, 0.06, 0.02),
         unite="(part de patients refusés)", base=dict(lam=2.4, mu=1.0, c=2, K=4),
         histoire="Des patients arrivent à 2,4 par heure, chaque médecin en soigne 1 par heure. "
                  "Le service a au plus 4 médecins et 8 places au total (soignés + en attente).",
         objectif="Taux de refus ≤ 15 % (★), ≤ 6 % (★★), ≤ 2 % (★★★) avec c ≤ 4 et K ≤ 8.",
         contrainte=lambda p: (p["c"] != INF and p["c"] <= 4 and p["K"] != INF and p["K"] <= 8, "c ≤ 4 et K ≤ 8"),
         indice="Agrandir la salle (K) aide, mais ajouter un médecin aide davantage."),
    dict(id="hotline", titre="La hotline impatiente", icon="hourglass", metrique="p_renege", op="<=",
         niveaux=(0.20, 0.12, 0.05), unite="(part d'appels abandonnés)",
         base=dict(lam=2.5, mu=1.0, c=2, patience=3.0),
         histoire="Les appelants raccrochent en moyenne après 3 minutes d'attente. Vous pouvez employer jusqu'à "
                  "4 opérateurs.",
         objectif="Abandons ≤ 20 % (★), ≤ 12 % (★★), ≤ 5 % (★★★) avec c ≤ 4.",
         contrainte=lambda p: (p["c"] != INF and p["c"] <= 4, "c ≤ 4 opérateurs"),
         indice="Comparez les modèles M/M/c avec et sans abandons."),
    dict(id="cafe", titre="Machine à café (source finie)", icon="clock", metrique="W", op="<=", niveaux=(4.0, 2.0, 1.4),
         unite="u.t. (temps de réponse)", base=dict(lam=0.2, mu=1.0, c=1, N=8),
         histoire="8 employés vont chacun chercher un café environ toutes les 5 minutes ; la machine sert en 1 minute "
                  "en moyenne. Vous pouvez installer une 2ᵉ machine.",
         objectif="Temps de réponse W ≤ 4 (★), ≤ 2 (★★), ≤ 1,4 (★★★) avec c ≤ 2.",
         contrainte=lambda p: (p["c"] != INF and p["c"] <= 2 and p["N"] != INF, "c ≤ 2, population finie conservée"),
         indice="Une population finie s'auto-régule : les clients déjà en file ne peuvent pas arriver."),
    dict(id="peage", titre="Le péage régulier", icon="door", metrique="Wq", op="<=", niveaux=(7.0, 3.5, 2.3),
         unite="u.t. (attente moyenne)", base=dict(lam=0.8, mu=1.0, c=1, srv="G", srv_forme=4.0),
         histoire="Une seule voie de péage, des véhicules à 0,8 par minute, des temps de passage très irréguliers "
                  "(cv² = 4). Vous ne pouvez pas ajouter de voie, mais vous pouvez standardiser le passage.",
         objectif="Attente Wq ≤ 7 (★), ≤ 3,5 (★★), ≤ 2,3 (★★★) avec c = 1 et le même débit moyen μ = 1.",
         contrainte=lambda p: (p["c"] == 1 and abs(p["mu"] - 1.0) < 1e-9, "c = 1 et μ = 1"),
         indice="La variabilité du service coûte cher : Pollaczek-Khinchine dit Wq ∝ (1 + cv²)."),
]


# ======================================================================
#  Aide
# ======================================================================
AIDE_TOPICS = [("demarrage", "Démarrage"), ("kendall", "Notation de Kendall"), ("modeles", "Modèles"),
               ("params", "Paramètres"), ("indic", "Indicateurs"), ("theorie", "Théorie"),
               ("etudes", "Études & défis"), ("clavier", "Raccourcis")]
AIDE_TITRES = {
    "demarrage": "Guide de démarrage", "kendall": "Notation de Kendall", "modeles": "Catalogue des modèles",
    "params": "Paramètres du modèle", "indic": "Indicateurs de performance", "theorie": "Formules et théorie",
    "etudes": "Études de sensibilité et défis", "clavier": "Raccourcis clavier",
}


def contenu_aide(topic):
    if topic == "demarrage":
        return [
            ("h1", "Guide de démarrage"),
            ("p", "Une file d'attente, c'est des clients qui arrivent, patientent éventuellement, sont servis puis "
                  "repartent. Cette application simule ce phénomène événement par événement et le compare aux formules "
                  "de la théorie des files."),
            ("h2", "En 5 étapes"),
            ("li", "1.  Écran « Modèle » : choisissez un modèle prêt à l'emploi (M/M/1, Erlang-B…) ou réglez vous-même "
                   "les lois d'arrivée et de service, le nombre de serveurs, la capacité…"),
            ("li", "2.  Écran « Simuler » (scène plein écran, forme des clients au choix) : appuyez sur le grand bouton rond. Les clients (des petits visages) arrivent, "
                   "font la queue, passent au guichet. Leur humeur se dégrade quand l'attente s'allonge."),
            ("li", "3.  Écran « Suivi » : lisez le journal et les indicateurs en direct : chaque arrivée, début de service, départ, refus ou abandon y est expliqué."),
            ("li", "4.  Écran « Résultats » : comparez les indicateurs simulés aux valeurs théoriques ; lancez une "
                   "analyse sur plusieurs réplications pour obtenir des intervalles de confiance."),
            ("li", "5.  Écran « Études » : balayez un paramètre (charge, serveurs, capacité) ou relevez les défis."),
            ("h2", "Unités"),
            ("p", "Le temps est exprimé en unités de temps (u.t.), à vous de choisir : minutes, secondes… "
                  "Les taux λ et μ sont des nombres de clients par u.t."),
            ("h2", "Info-bulles"),
            ("p", "Laissez la souris immobile sur n'importe quel bouton, curseur, interrupteur, puce, graphique, "
                  "ligne de tableau ou machine : une info-bulle explique son rôle."),
        ]
    if topic == "kendall":
        return [
            ("h1", "Notation de Kendall"),
            ("p", "Un modèle de file s'écrit  A / B / c / K / N / D  :"),
            ("li", "A :  loi des intervalles entre arrivées.  B :  loi des durées de service."),
            ("li", "c :  nombre de serveurs identiques (∞ possible).  K :  capacité totale du système (clients en "
                   "service + en attente) ; ∞ si illimitée."),
            ("li", "N :  taille de la population de clients (∞ = source infinie).  D :  discipline de service."),
            ("p", "Les trois derniers champs sont omis quand ils valent ∞, ∞ et FIFO."),
            ("h2", "Lettres des lois"),
            ("li", "M :  exponentielle (« Markov », sans mémoire) ; cv² = 1."),
            ("li", "D :  déterministe (constante) ; cv² = 0."),
            ("li", "Eₖ :  Erlang d'ordre k (somme de k exponentielles) ; cv² = 1/k, plus régulière que M."),
            ("li", "H₂ :  hyperexponentielle à 2 phases ; cv² > 1, très irrégulière."),
            ("li", "U :  uniforme sur [0 ; 2·moyenne] ; cv² = 1/3."),
            ("li", "LN, G :  log-normale (G = « générale » avec cv² au choix)."),
            ("h2", "Exemples"),
            ("code", "M/M/1       un guichet, tout markovien, file illimitée\n"
                     "M/M/3/10    3 serveurs, au plus 10 clients dans le système\n"
                     "M/D/1       service constant\n"
                     "M/M/1/∞/6   population finie de 6 clients\n"
                     "M/G/1/∞/∞/SJF   plus court service d'abord"),
        ]
    if topic == "modeles":
        out = [("h1", "Catalogue des modèles"),
               ("p", "L'écran Modèle propose les modèles suivants. Cliquer sur une puce charge ses paramètres ; "
                     "vous pouvez ensuite les modifier librement (la puce se désélectionne : modèle personnalisé).")]
        for nom, desc, _ in PRESETS:
            out += [("h2", nom), ("p", desc)]
        out += [("p", "Au-delà du catalogue, toute combinaison de lois (M, D, Eₖ, H₂, U, LN, G), de serveurs, de "
                      "capacité, de population, de discipline, d'abandons et de refus est possible.")]
        return out
    if topic == "params":
        return [
            ("h1", "Paramètres du modèle"),
            ("h2", "Arrivées et service"),
            ("li", "Taux λ (arrivées) et μ (service) : intervalle moyen = 1/λ, durée moyenne de service = 1/μ."),
            ("li", "Forme : k pour Erlang, cv² pour H₂, log-normale et générale. Ignoré pour M, D et U."),
            ("h2", "Structure"),
            ("li", "Serveurs c : de 1 à 10, ou ∞ (interrupteur)."),
            ("li", "Capacité K : nombre maximum de clients dans le système. À l'arrivée, si le système contient déjà K "
                   "clients, le nouveau est refusé (blocage)."),
            ("li", "Population N : si elle est finie, chaque client revient après un temps de réflexion de moyenne 1/λ "
                   "(λ est alors un taux par client)."),
            ("li", "Discipline : FIFO, LIFO, SIRO (aléatoire) ou SJF (plus court d'abord)."),
            ("h2", "Comportements"),
            ("li", "Patience : un client en file abandonne après un délai exponentiel de moyenne choisie (reneging)."),
            ("li", "Refus : probabilité qu'un client renonce à entrer sans même essayer (balking)."),
            ("h2", "Simulation"),
            ("li", "Horizon : durée simulée. Chauffe : durée initiale ignorée dans les statistiques (régime transitoire)."),
            ("li", "Graine : 0 = aléatoire ; une valeur fixe rend la simulation reproductible."),
            ("h2", "Santé du modèle"),
            ("p", "La jauge indique la charge ρ = λ / (c·μ). Au-delà de 100 % le système est instable : la file "
                  "grossit indéfiniment, sauf si une capacité finie, des abandons ou une population finie la bornent."),
        ]
    if topic == "indic":
        return [
            ("h1", "Indicateurs de performance"),
            ("li", "L :  nombre moyen de clients dans le système (en attente + en service)."),
            ("li", "Lq :  nombre moyen de clients en attente."),
            ("li", "W :  temps moyen passé dans le système.  Wq :  temps moyen d'attente avant le service."),
            ("li", "Utilisation :  fraction du temps où un serveur est occupé (nombre moyen de serveurs occupés "
                   "si c = ∞)."),
            ("li", "Débit accepté λ_eff :  clients réellement admis par unité de temps."),
            ("li", "P(blocage) :  proportion d'arrivées refusées parce que le système est plein."),
            ("li", "P(abandon) :  proportion de clients admis qui quittent la file par impatience."),
            ("p", "W et Wq incluent les clients qui abandonnent (leur attente avant de partir)."),
            ("h2", "Intervalle de confiance"),
            ("p", "Avec plusieurs réplications indépendantes, on affiche moyenne ± demi-largeur de l'IC à 95 % "
                  "(loi de Student)."),
            ("h2", "Verdict"),
            ("li", "Coche verte : la simulation est compatible avec la théorie (dans l'IC ou à moins de 3 %)."),
            ("li", "≈ orange : la théorie est une approximation ou l'écart reste modéré."),
            ("li", "Croix rouge : écart important (horizon trop court ? chauffe insuffisante ?)."),
        ]
    if topic == "theorie":
        return [
            ("h1", "Formules et théorie"),
            ("h2", "Loi de Little"),
            ("code", "L = λ_eff · W        Lq = λ_eff · Wq        W = Wq + 1/μ"),
            ("h2", "Processus de naissance et de mort (exact)"),
            ("p", "M/M/c/K, M/M/c, M/M/1//N, M/M/∞, avec abandons exponentiels et refus aléatoires : les probabilités "
                  "d'état Pₙ se déduisent des taux λₙ (naissance) et μₙ = min(n, c)·μ + max(n − c, 0)·θ (mort)."),
            ("code", "Pₙ = P₀ · Π (λᵢ / μᵢ₊₁)      avec  Σ Pₙ = 1"),
            ("h2", "M/G/1 : Pollaczek-Khinchine (exact)"),
            ("code", "Wq = λ · E[S²] / (2 (1 − ρ))  =  ρ · (1 + cs²) / (2 μ (1 − ρ))"),
            ("p", "L'attente croît linéairement avec la variabilité du service (cs²)."),
            ("h2", "G/M/1 (exact)"),
            ("p", "Pour D, Eₖ, H₂, U en arrivées : σ est la racine de σ = A*(μ(1 − σ)), puis Wq = σ / (μ(1 − σ))."),
            ("h2", "M/G/1/K (exact)"),
            ("p", "Chaîne de Markov incluse aux instants de départ ; les probabilités temporelles se déduisent par "
                  "pₙ = πₙ / (π₀ + ρ). Disponible pour un service M, D, Eₖ ou H₂."),
            ("h2", "G/G/1 et G/G/c (approximations)"),
            ("code", "Kingman (c = 1) :  Wq ≈ ρ/(1 − ρ) · (ca² + cs²)/2 · E[S]\n"
                     "Allen-Cunneen   :  Wq ≈ Wq(M/M/c) · (ca² + cs²)/2"),
            ("h2", "D/D/c"),
            ("p", "Si λ ≤ c·μ, aucune attente. Sinon le système sature et refuse une fraction 1 − c·μ/λ des clients."),
            ("h2", "Quand la théorie manque"),
            ("p", "Pour les autres combinaisons (capacité finie avec lois générales, SJF, population finie "
                  "non markovienne…), la simulation fait foi : la colonne « Théorie » reste vide."),
        ]
    if topic == "etudes":
        return [
            ("h1", "Études de sensibilité et défis"),
            ("h2", "Balayage"),
            ("p", "Fait varier un paramètre (λ, nombre de serveurs c ou capacité K) et trace l'indicateur choisi. "
                  "Chaque point est la moyenne de plusieurs réplications avec barres d'erreur ; la courbe théorique "
                  "pointillée s'ajoute quand elle existe. Vous voyez l'explosion de l'attente quand ρ → 1."),
            ("h2", "Défis"),
            ("p", "Cinq situations concrètes. Chargez un défi, réglez le modèle, puis « Tester ma solution » : "
                  "8 simulations sont lancées et le résultat est comparé à trois paliers (★ ★★ ★★★). "
                  "Une contrainte de budget (par exemple c ≤ 6) doit être respectée."),
        ]
    return [
        ("h1", "Raccourcis clavier"),
        ("li", "Espace             Lecture / Pause de la simulation"),
        ("li", "S  ou  →           Événement suivant (pas à pas)"),
        ("li", "F                  Fin rapide jusqu'à l'horizon"),
        ("li", "R                  Nouvelle simulation"),
        ("li", "Échap              Arrêter et remettre à zéro"),
        ("li", "1  2  3  4  5  6   Modèle · Simuler · Suivi · Résultats · Études · Aide"),
        ("li", "Ctrl + E           Exporter les résultats en CSV"),
        ("li", "Ctrl + S           Enregistrer le journal"),
        ("li", "F1                 Ouvrir ce guide"),
        ("p", "Les raccourcis de lecture (Espace, S, F, R, Échap) fonctionnent depuis les écrans Simuler et Suivi."),
    ]


# ======================================================================
#  Application
# ======================================================================
def decrire_modele(p):
    c = "∞" if p["c"] == INF else int(p["c"])
    K = "illimitée" if p["K"] == INF else int(p["K"])
    N = "infinie" if p["N"] == INF else int(p["N"])
    l1 = (f"Arrivées : {DIST_LIB[p['arr']]}, λ = {p['lam']:g}"
          + (f", forme {p['arr_forme']:g}" if p["arr"] in ("E", "H", "LN", "G") else ""))
    l2 = (f"Service : {DIST_LIB[p['srv']]}, μ = {p['mu']:g}"
          + (f", forme {p['srv_forme']:g}" if p["srv"] in ("E", "H", "LN", "G") else ""))
    l3 = f"{c} serveur(s) · capacité {K} · population {N} · discipline {p['disc']}"
    extra = []
    if p["patience"]:
        extra.append(f"abandons (patience moyenne {p['patience']:g})")
    if p["balk"] > 0:
        extra.append(f"refus aléatoires {p['balk'] * 100:.0f} %")
    if extra:
        l3 += " · " + ", ".join(extra)
    return l1 + "\n" + l2 + "\n" + l3


class App:
    PAGES = [("modele", "Modèle", "model", "Choisir et régler le modèle de file (touche 1)"),
             ("sim", "Simuler", "queue", "Simulation visuelle plein écran de la file (touche 2)"),
             ("suivi", "Suivi", "list", "Indicateurs en direct, courbe n(t) et journal (touche 3)"),
             ("res", "Résultats", "stats", "Théorie contre simulation, distributions (touche 4)"),
             ("etudes", "Études", "flask", "Balayage de paramètres et défis (touche 5)"),
             ("aide", "Aide", "book", "Guide, notation de Kendall, formules (touche 6)")]
    SOUS_TITRES = {"modele": "Choix et réglage du modèle", "sim": "Simulation visuelle", "suivi": "Indicateurs, courbe et journal",
                   "res": "Théorie contre simulation", "etudes": "Balayages et défis", "aide": "Centre d'aide"}
    MAX_LIGNES = 3000

    def __init__(self, root):
        self.root = root
        root.title("Files d'attente — Laboratoire de simulation")
        root.configure(bg=T.BG)
        self._sync = False
        self.p = params_preset({})
        self.th = None
        self.sim = None
        self.running = False
        self.ff = False
        self.clock = 0.0
        self.last = time.perf_counter()
        self.effets, self.mouv, self.pos = [], {}, {}
        self.cum = dict(arr=0, dep=0, block=0, balk=0, renege=0)
        self.geo = {}
        self.res = None
        self.occupe = False
        self.etoiles = {}
        self.defi_actif = None
        self.page = None
        self.forme = "visage"
        self.rcl = 13
        self.pages = {}
        self.ctl = {}
        self.status = tk.StringVar(value="")
        self.sv = {}
        self._toast_job = None
        self._live = 0.0
        self.bal_var, self.bal_met = "lam", "W"

        self.construire_coque()
        self.construire_modele()
        self.construire_sim()
        self.construire_suivi()
        self.construire_res()
        self.construire_etudes()
        self.construire_aide()
        self.toast_lbl = tk.Label(root, bg=T.TIP, fg="#FFFFFF", font=fnt(10), padx=18, pady=10)
        self.lier_raccourcis()
        self.appliquer_preset("M/M/1")
        self.afficher("modele")
        self.boucle()

    # ------------------------------------------------------------------
    #  Coque
    # ------------------------------------------------------------------
    def construire_coque(self):
        r = self.root
        barre = tk.Frame(r, bg=T.CARD, height=68)
        barre.pack(fill="x")
        barre.pack_propagate(False)
        tk.Frame(r, bg=T.BORDER, height=1).pack(fill="x")
        logo = tk.Canvas(barre, width=44, height=44, bg=T.CARD, highlightthickness=0)
        logo.pack(side="left", padx=(18, 12), pady=12)
        draw_logo(logo, 44)
        TIP.attach(logo, "Files d'attente — laboratoire de simulation")
        txt = tk.Frame(barre, bg=T.CARD)
        txt.pack(side="left")
        tk.Label(txt, text="Files d'attente", font=fnt(15, "bold"), fg=T.TEXT, bg=T.CARD).pack(anchor="w")
        self.sous_titre = tk.Label(txt, text="", font=fnt(9), fg=T.MUTED, bg=T.CARD)
        self.sous_titre.pack(anchor="w")
        self.btn_more = IconButton(barre, "more", self.menu_aide, "Menu d'aide : guide, Kendall, formules, à propos",
                                   size=42, kind="light")
        self.btn_more.pack(side="right", padx=(0, 12))
        IconButton(barre, "help", lambda: self.ouvrir_aide("demarrage"), "Ouvrir le guide de démarrage (F1)",
                   size=42, kind="light").pack(side="right")
        self.lbl_kendall = tk.Label(barre, text="", font=fnt(12, "bold"), fg=T.PRIMARY, bg=T.PRIMARY_L, padx=14, pady=5)
        self.lbl_kendall.pack(side="right", padx=14)
        TIP.attach(self.lbl_kendall, "Notation de Kendall du modèle courant : A/B/c/K/N/discipline")

        nav_zone = tk.Frame(r, bg=T.CARD)
        nav_zone.pack(side="bottom", fill="x")
        tk.Frame(nav_zone, bg=T.BORDER, height=1).pack(fill="x")
        nav = tk.Frame(nav_zone, bg=T.CARD)
        nav.pack(fill="x")
        self.nav = {}
        for i, (cle, lib, ico, tip) in enumerate(self.PAGES):
            nav.columnconfigure(i, weight=1, uniform="nav")
            it = NavItem(nav, ico, lib, lambda k=cle: self.afficher(k), tip)
            it.grid(row=0, column=i, sticky="ew")
            self.nav[cle] = it
        self.conteneur = tk.Frame(r, bg=T.BG)
        self.conteneur.pack(fill="both", expand=True)

    def afficher(self, cle):
        if self.page:
            self.pages[self.page].pack_forget()
        self.page = cle
        self.pages[cle].pack(fill="both", expand=True)
        for k, it in self.nav.items():
            it.set_active(k == cle)
        self.sous_titre.config(text=self.SOUS_TITRES[cle])
        TIP.cacher()
        self.root.after(40, self.rafraichir_page)

    def rafraichir_page(self):
        if self.page == "sim":
            self.dessiner_scene()
        elif self.page == "suivi":
            self.maj_live(True)
            self.maj_traj()
        elif self.page == "res":
            self.maj_resultats()
        elif self.page == "etudes":
            self.maj_balayage_vide()
        elif self.page == "modele":
            self.dessiner_sante()

    def molette(self, e):
        pg = self.pages.get(self.page)
        if hasattr(pg, "scroll"):
            d = -1 if (getattr(e, "num", 0) == 4 or getattr(e, "delta", 0) > 0) else 1
            pg.scroll(2 * d)

    # ------------------------------------------------------------------
    #  Page Modèle
    # ------------------------------------------------------------------
    def construire_modele(self):
        sf = self.pages["modele"] = ScrollFrame(self.conteneur)
        inn = sf.inner
        for i in range(3):
            inn.columnconfigure(i, weight=1, uniform="m")

        cm = Card(inn, pad=14)
        cm.grid(row=0, column=0, columnspan=3, sticky="ew", padx=14, pady=(14, 6))
        b = cm.body
        tete = tk.Frame(b, bg=T.CARD)
        tete.pack(fill="x")
        self.badge = tk.Canvas(tete, width=250, height=70, bg=T.CARD, highlightthickness=0)
        self.badge.pack(side="left")
        TIP.attach(self.badge, "Notation de Kendall : A/B/c/K/N/discipline (voir Aide > Notation de Kendall)")
        self.lbl_desc = tk.Label(tete, text="", font=fnt(10), fg=T.MUTED, bg=T.CARD, justify="left", anchor="w")
        self.lbl_desc.pack(side="left", padx=16, fill="x", expand=True)
        self.lbl_defi = tk.Label(b, text="", font=fnt(9, "bold"), fg="#B45309", bg=mix(T.ORANGE, "#FFFFFF", .85),
                                 padx=10, pady=5, anchor="w", justify="left")
        tk.Label(b, text="Modèles prêts à l'emploi", font=fnt(9, "bold"), fg=T.MUTED, bg=T.CARD).pack(anchor="w", pady=(10, 4))
        self.chips_preset = ChipGroup(b, [(n, n) for n, _, _ in PRESETS], None, self.appliquer_preset,
                                      {n: n + " — " + d for n, d, _ in PRESETS})
        self.chips_preset.pack(fill="x")

        # --- arrivées / service / structure
        ca = Card(inn, pad=14)
        ca.grid(row=1, column=0, sticky="nsew", padx=(14, 6), pady=6)
        self.construire_loi(ca.body, "arr")
        cs = Card(inn, pad=14)
        cs.grid(row=1, column=1, sticky="nsew", padx=6, pady=6)
        self.construire_loi(cs.body, "srv")
        cst = Card(inn, pad=14)
        cst.grid(row=1, column=2, sticky="nsew", padx=(6, 14), pady=6)
        self.construire_structure(cst.body)

        # --- comportements + simulation
        cc = Card(inn, pad=14)
        cc.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=(14, 6), pady=(6, 14))
        self.construire_options(cc.body)
        csn = Card(inn, pad=14)
        csn.grid(row=2, column=2, sticky="nsew", padx=(6, 14), pady=(6, 14))
        self.construire_sante(csn.body)

    def titre_carte(self, parent, icone, titre, sous=""):
        h = tk.Frame(parent, bg=T.CARD)
        h.pack(fill="x", pady=(0, 8))
        ic = tk.Canvas(h, width=30, height=30, bg=T.CARD, highlightthickness=0)
        ic.pack(side="left")
        ic.create_oval(1, 1, 29, 29, fill=T.PRIMARY_L, outline="")
        draw_icon(ic, icone, 15, 15, 17, T.PRIMARY)
        tk.Label(h, text=titre, font=fnt(11, "bold"), fg=T.TEXT, bg=T.CARD).pack(side="left", padx=8)
        if sous:
            tip_all(h, sous)

    def construire_loi(self, parent, cote):
        arr = cote == "arr"
        self.titre_carte(parent, "users" if arr else "gauge", "Arrivées" if arr else "Service",
                         "Loi des intervalles entre arrivées successives." if arr else
                         "Loi de la durée de service d'un client.")
        items = [(k, DIST_COURT[k] if k != "E" else "Eₖ") for k in DIST_LIB]
        tips = {
            "M": "M — exponentielle : sans mémoire, cv² = 1. Arrivées de Poisson / service Markovien.",
            "D": "D — déterministe : durée constante, cv² = 0 (cadence parfaite).",
            "E": "Eₖ — Erlang d'ordre k : somme de k exponentielles, cv² = 1/k (plus régulière).",
            "H": "H₂ — hyperexponentielle : mélange de 2 exponentielles, cv² > 1 (très irrégulière).",
            "U": "U — uniforme sur [0 ; 2·moyenne], cv² = 1/3.",
            "LN": "LN — log-normale de moyenne et cv² donnés.",
            "G": "G — loi générale (log-normale) : choisissez la variabilité avec cv².",
        }
        chips = ChipGroup(parent, items, self.p["arr" if arr else "srv"],
                          lambda code, c=cote: self.choisir_loi(c, code), tips)
        chips.pack(fill="x", pady=(0, 6))
        cle = "lam" if arr else "mu"
        r = Reglage(self, parent, "Taux λ" if arr else "Taux μ", 0.05, 5.0, self.p[cle], 0.05,
                    lambda v: f"{v:.2f}", lambda v, k=cle: self.set_param(k, v),
                    "Arrivées : clients par unité de temps (intervalle moyen = 1/λ)." if arr else
                    "Service : clients servis par unité de temps par un serveur (durée moyenne = 1/μ).",
                    "bolt")
        r.frame.pack(fill="x", pady=(4, 0))
        f = Reglage(self, parent, "Forme", 1, 10, 2, 1, lambda v: f"{v:g}",
                    lambda v, c=cote: self.set_forme(c, v),
                    "Paramètre de forme : k pour Erlang, variabilité cv² pour H₂, LN et G.", "chart")
        f.frame.pack(fill="x", pady=(4, 0))
        info = tk.Label(parent, text="", font=fnt(9), fg=T.MUTED, bg=T.CARD, anchor="w", justify="left")
        info.pack(fill="x", pady=(6, 0))
        self.ctl[cote] = dict(chips=chips, rate=r, forme=f, info=info)

    def choisir_loi(self, cote, code):
        k = "arr" if cote == "arr" else "srv"
        self.p[k] = code
        defauts = {"E": 2.0, "H": 4.0, "LN": 1.5, "G": 1.5}
        if code in defauts:
            self.p[k + "_forme"] = defauts[code]
        self.cfg_forme(cote)
        self.maj_modele()

    def set_forme(self, cote, v):
        self.p["arr_forme" if cote == "arr" else "srv_forme"] = float(v)
        self.maj_modele()

    def cfg_forme(self, cote):
        k = "arr" if cote == "arr" else "srv"
        code, val = self.p[k], self.p[k + "_forme"]
        f = self.ctl[cote]["forme"]
        s = f.slider
        if code == "E":
            s.regler(1, 10, 1, int(round(val)))
            f.titre.config(text="Phases k")
            f.fmt = lambda v: f"{int(v)}"
            f.enable(True)
        elif code == "H":
            s.regler(1.2, 10.0, 0.1, val if val > 1.2 else 4.0)
            f.titre.config(text="Variabilité cv²")
            f.fmt = lambda v: f"{v:.1f}"
            f.enable(True)
        elif code in ("LN", "G"):
            s.regler(0.1, 6.0, 0.1, val)
            f.titre.config(text="Variabilité cv²")
            f.fmt = lambda v: f"{v:.1f}"
            f.enable(True)
        else:
            f.titre.config(text="Forme (sans objet)")
            f.fmt = lambda v: "—"
            f.enable(False)
        f.label.config(text=f.fmt(s.get()) if f.slider.enabled else "—")
        self.p[k + "_forme"] = float(s.get()) if f.slider.enabled else self.p[k + "_forme"]

    def construire_structure(self, parent):
        self.titre_carte(parent, "layers", "Structure", "Serveurs, capacité du système, population de clients et discipline.")
        r = Reglage(self, parent, "Serveurs c", 1, 10, 1, 1, lambda v: f"{int(v)}", lambda v: self.set_param("c", v),
                    "Nombre de serveurs identiques travaillant en parallèle (∞ = libre-service).", "users", inf=True)
        r.frame.pack(fill="x")
        k = Reglage(self, parent, "Capacité K", 1, 50, 10, 1, lambda v: f"{int(v)}", lambda v: self.set_param("K", v),
                    "Nombre maximal de clients dans le système (en service + en attente). Au-delà : refus.",
                    "door", inf=True)
        k.frame.pack(fill="x", pady=(6, 0))
        n = Reglage(self, parent, "Population N", 1, 50, 6, 1, lambda v: f"{int(v)}", lambda v: self.set_param("N", v),
                    "Source finie : N clients au total ; chacun revient après un temps de réflexion de moyenne 1/λ.",
                    "users", inf=True)
        n.frame.pack(fill="x", pady=(6, 0))
        tk.Label(parent, text="Discipline", font=fnt(9, "bold"), fg=T.TEXT, bg=T.CARD).pack(anchor="w", pady=(8, 4))
        disc = ChipGroup(parent, [(d, d) for d in DISCIPLINES], self.p["disc"],
                         lambda d: self.set_param("disc", d), dict(DISCIPLINES))
        disc.pack(fill="x")
        self.ctl.update(c=r, K=k, N=n, disc=disc)

    def construire_options(self, parent):
        self.titre_carte(parent, "model", "Comportements et simulation",
                         "Impatience, refus aléatoires et paramètres de la simulation.")
        g = tk.Frame(parent, bg=T.CARD)
        g.pack(fill="x")
        g.columnconfigure(0, weight=1, uniform="o")
        g.columnconfigure(1, weight=1, uniform="o")
        self.pat_var = tk.BooleanVar(value=False)
        pat = Reglage(self, g, "Patience moyenne", 0.5, 20.0, 4.0, 0.5, lambda v: f"{v:g}", self.set_patience,
                      "Durée moyenne (exponentielle) avant qu'un client en file abandonne.", "hourglass")
        pat.frame.grid(row=0, column=0, sticky="ew", padx=(0, 14))
        tg = Toggle(pat.frame.winfo_children()[0], self.pat_var,
                    "Activer les abandons : les clients impatients quittent la file (reneging).")
        tg.pack(side="right", padx=(0, 6))
        self.pat_var.trace_add("write", lambda *a: self.set_patience(pat.slider.get()))
        bk = Reglage(self, g, "Refus aléatoire", 0.0, 0.9, 0.0, 0.05, lambda v: f"{v * 100:.0f} %",
                     lambda v: self.set_param("balk", v),
                     "Probabilité qu'un client renonce à entrer sans essayer (balking).", "ban")
        bk.frame.grid(row=0, column=1, sticky="ew")
        hz = Reglage(self, g, "Horizon", 100, 10000, 2000, 100, lambda v: f"{int(v)} u.t.",
                     lambda v: self.set_param("horizon", float(v)),
                     "Durée simulée en unités de temps. Plus long = résultats plus précis.", "clock")
        hz.frame.grid(row=1, column=0, sticky="ew", padx=(0, 14), pady=(8, 0))
        wu = Reglage(self, g, "Chauffe (ignorée)", 0, 2000, 0, 50, lambda v: f"{int(v)} u.t.",
                     lambda v: self.set_param("warmup", float(v)),
                     "Début de simulation exclu des statistiques (régime transitoire).", "layers")
        wu.frame.grid(row=1, column=1, sticky="ew", pady=(8, 0))
        sd = Reglage(self, g, "Graine aléatoire", 0, 999, 0, 1, lambda v: "aléatoire" if v == 0 else str(int(v)),
                     lambda v: self.set_param("seed", int(v)),
                     "0 = tirage différent à chaque fois ; une valeur fixe rend la simulation reproductible.", "dice")
        sd.frame.grid(row=2, column=0, sticky="ew", padx=(0, 14), pady=(8, 0))
        de = IconButton(g, "dice", lambda: self.ctl["seed"].set(random.randint(1, 999)) or
                        self.set_param("seed", self.ctl["seed"].slider.get()),
                        "Tirer une graine au hasard (la simulation devient reproductible)", size=40)
        de.grid(row=2, column=1, sticky="w", pady=(8, 0))
        self.ctl.update(patience=pat, balk=bk, horizon=hz, warmup=wu, seed=sd)

    def set_patience(self, v):
        if self._sync:
            return
        self.p["patience"] = float(v) if self.pat_var.get() else None
        self.ctl["patience"].enable(self.pat_var.get())
        self.maj_modele()

    def construire_sante(self, parent):
        self.titre_carte(parent, "gauge", "Santé du modèle", "Charge du système et solution analytique disponible.")
        self.sante = tk.Canvas(parent, height=96, bg=T.CARD, highlightthickness=0)
        self.sante.pack(fill="x")
        self.sante.bind("<Configure>", lambda e: self.dessiner_sante())
        TIP.attach(self.sante, "Charge ρ = λ / (c·μ). Vert < 70 %, orange jusqu'à 100 %, rouge au-delà (instable).")
        self.lbl_th = tk.Label(parent, text="", font=fnt(9, "bold"), padx=10, pady=4, anchor="w")
        self.lbl_th.pack(anchor="w", pady=(6, 4))
        self.lbl_note = tk.Label(parent, text="", font=fnt(9), fg=T.MUTED, bg=T.CARD, justify="left", anchor="w",
                                 wraplength=300)
        self.lbl_note.pack(fill="x")
        TIP.attach(self.lbl_th, "Disponibilité d'une formule analytique pour ce modèle (voir Aide > Formules).")

    def set_param(self, k, v):
        if self._sync:
            return
        self.p[k] = v
        self.maj_modele()

    def appliquer_preset(self, nom):
        over = next(o for n, _, o in PRESETS if n == nom)
        keep = {k: self.p[k] for k in ("horizon", "warmup", "seed")}
        self.p = params_preset(over)
        self.p.update(keep)
        self.sync_widgets()
        self.maj_modele(preset=nom)

    def sync_widgets(self):
        self._sync = True
        p = self.p
        for cote, (code, rate, forme) in (("arr", ("arr", "lam", "arr_forme")), ("srv", ("srv", "mu", "srv_forme"))):
            ctl = self.ctl[cote]
            ctl["chips"].set(p[code])
            ctl["rate"].set(p[rate])
            self.cfg_forme(cote)
            ctl["forme"].slider.set(p[forme]) if ctl["forme"].slider.enabled else None
            ctl["forme"].label.config(text=ctl["forme"].fmt(ctl["forme"].slider.get()) if ctl["forme"].slider.enabled else "—")
        for k in ("c", "K", "N"):
            self.ctl[k].set(p[k])
        self.ctl["disc"].set(p["disc"])
        self.pat_var.set(bool(p["patience"]))
        self.ctl["patience"].set(p["patience"] or 4.0)
        self.ctl["patience"].enable(bool(p["patience"]))
        for k in ("balk", "horizon", "warmup", "seed"):
            self.ctl[k].set(p[k])
        self._sync = False

    def maj_modele(self, preset=None):
        avant = (self.p["c"], self.p["K"])
        self.p = normaliser(self.p)
        if (self.p["c"], self.p["K"]) != avant:
            self.sync_widgets()
        self.th = theorie(self.p)
        self.chips_preset.set(preset)
        kd = kendall(self.p)
        self.lbl_kendall.config(text=kd)
        c = self.badge
        c.delete("all")
        rrect(c, 2, 2, 248, 68, 22, fill=T.PRIMARY_L, outline="")
        taille = 24
        while taille > 12 and mesure(kd, fnt(taille, "bold")) > 220:
            taille -= 2
        c.create_text(125, 35, text=kd, font=fnt(taille, "bold"), fill=T.PRIMARY_D)
        self.lbl_desc.config(text=decrire_modele(self.p))
        self.ctl["arr"]["info"].config(text=self.info_loi("arr"))
        self.ctl["srv"]["info"].config(text=self.info_loi("srv"))
        self.dessiner_sante()
        if hasattr(self, "scene"):
            self.nouvelle_sim()

    def info_loi(self, cote):
        k = "arr" if cote == "arr" else "srv"
        taux = self.p["lam" if cote == "arr" else "mu"]
        return f"moyenne {1 / taux:.2f} u.t. · cv² = {scv(self.p[k], self.p[k + '_forme']):.2f}"

    def dessiner_sante(self):
        if not hasattr(self, "sante"):
            return
        c = self.sante
        c.delete("all")
        w, h = dims(c)
        p, th = self.p, self.th
        rho = charge(p)
        if rho is None and th and th.get("util") is not None and p["c"] != INF:
            rho = th["util"]
        x0, x1, y = 14, w - 14, 46
        sc = (x1 - x0) / 1.5
        for a, b, col in ((0, .7, T.GREEN), (.7, 1.0, T.ORANGE), (1.0, 1.5, T.RED)):
            c.create_rectangle(x0 + sc * a, y - 6, x0 + sc * b, y + 6, fill=mix(col, "#FFFFFF", .35), outline="")
        for v, lab in ((0, "0"), (.7, "70 %"), (1.0, "100 %"), (1.5, "150 %")):
            c.create_text(x0 + sc * v, y + 17, text=lab, font=fnt(7), fill=T.MUTED)
        if rho is None or p["c"] == INF:
            msg = "∞ serveurs : aucune saturation possible" if p["c"] == INF else "Charge non définie"
            c.create_text(w / 2, 20, text=msg, font=fnt(10, "bold"), fill=T.TEXT)
            return
        mx = x0 + sc * min(rho, 1.5)
        c.create_polygon(mx - 7, y - 20, mx + 7, y - 20, mx, y - 7, fill=T.TEXT, outline="")
        c.create_text(x0, 14, text=f"ρ = {rho:.2f}", anchor="w", font=fnt(12, "bold"), fill=T.TEXT)
        if rho < 0.5:
            m = "charge légère"
        elif rho < 0.8:
            m = "charge normale"
        elif rho < 1:
            m = "charge élevée : l'attente explose près de 100 %"
        elif p["K"] != INF or p["patience"] or p["N"] != INF:
            m = "surcharge, mais bornée (capacité, abandons ou population)"
        else:
            m = "SURCHARGE : file instable"
        c.create_text(x1, 14, text=m, anchor="e", font=fnt(8), fill=T.MUTED)
        n = th["nature"] if th else None
        col, txt = {"exacte": (T.GREEN, "Théorie : solution exacte"), "approchée": (T.ORANGE, "Théorie : approximation"),
                    None: (T.MUTED, "Théorie : indisponible")}[n]
        self.lbl_th.config(text=txt, fg="#FFFFFF", bg=col)
        self.lbl_note.config(text=th["note"] if th else "")

    # ------------------------------------------------------------------
    #  Page Simuler
    # ------------------------------------------------------------------
    FORMES = [("visage", "Visages"), ("disque", "Disques"), ("carre", "Paquets Internet"),
              ("voiture", "Voitures"), ("bonhomme", "Bonhommes"),
              ("etoile", "Étoiles")]

    def construire_sim(self):
        p = self.pages["sim"] = tk.Frame(self.conteneur, bg=T.BG)
        haut = tk.Frame(p, bg=T.BG)
        haut.pack(side="top", fill="x", padx=16, pady=(12, 6))
        tk.Label(haut, text="Forme des clients", font=fnt(9, "bold"), fg=T.MUTED, bg=T.BG).pack(side="left", padx=(2, 10))
        self.chips_forme = ChipGroup(haut, self.FORMES, self.forme, self.set_forme_client,
                                     {"visage": "Visages dont l'humeur se dégrade avec l'attente.",
                                      "disque": "Disques colorés avec un contour noir.",
                                      "carre": "Paquets réseau qui transitent dans la file et les serveurs.",
                                      "voiture": "Voitures détaillées (parfait pour un péage ou un garage).",
                                      "bonhomme": "Silhouettes masculines et féminines.", "etoile": "Étoiles."})
        self.chips_forme.pack(side="left", fill="x", expand=True)

        dock = Card(p, pad=10, radius=26)
        dock.pack(side="bottom", fill="x", padx=16, pady=(4, 14))
        tk.Label(p, textvariable=self.status, font=fnt(9), fg=T.MUTED, bg=T.BG, anchor="w").pack(side="bottom", fill="x", padx=22)
        row = tk.Frame(dock.body, bg=T.CARD)
        row.pack(fill="x")
        bt = tk.Frame(row, bg=T.CARD)
        bt.pack(side="left")
        self.b_new = IconButton(bt, "reset", self.nouvelle_sim_btn, "Nouvelle simulation : remet l'horloge à zéro (R)",
                                size=48, label="Remise à zéro")
        self.fab = IconButton(bt, "play", self.fab_action, "Lancer la simulation (Espace)", size=62, kind="filled",
                              label="Lecture", icon_size=30, fab=True)
        self.b_step = IconButton(bt, "step", self.pas_sim, "Événement suivant : traite un seul événement (S ou →)",
                                 size=48, label="Pas à pas")
        self.b_ff = IconButton(bt, "ff", self.fin_rapide, "Fin rapide : simule jusqu'à l'horizon sans animation (F)",
                               size=48, label="Fin rapide")
        for w in (self.b_new, self.fab, self.b_step, self.b_ff):
            w.pack(side="left", padx=12, anchor="s")
        rg = tk.Frame(row, bg=T.CARD)
        rg.pack(side="left", fill="x", expand=True, padx=(28, 10))
        self.vit = Reglage(self, rg, "Vitesse", 1, 100, 40, 1, lambda v: f"{self.tu_par_s(v):.1f} u.t./s", lambda v: None,
                           "Temps simulé écoulé par seconde réelle (échelle logarithmique).", "bolt")
        self.vit.frame.pack(fill="x")

        self.scene = tk.Canvas(p, bg=T.BG, highlightthickness=0)
        self.scene.pack(fill="both", expand=True, padx=16, pady=(0, 8))
        self.scene.bind("<Configure>", lambda e: self.dessiner_scene())
        TIP.attach(self.scene, self.tip_scene, delay=350)

    def set_forme_client(self, cle):
        self.forme = cle
        self.dessiner_scene()

    def construire_suivi(self):
        p = self.pages["suivi"] = tk.Frame(self.conteneur, bg=T.BG)
        p.columnconfigure(0, weight=1, uniform="s")
        p.columnconfigure(1, weight=1, uniform="s")
        p.rowconfigure(0, weight=1)
        d = tk.Frame(p, bg=T.BG)
        d.grid(row=0, column=0, sticky="nsew", padx=(14, 7), pady=14)
        dj = tk.Frame(p, bg=T.BG)
        dj.grid(row=0, column=1, sticky="nsew", padx=(7, 14), pady=14)

        # colonne droite : KPI, courbe, journal
        kp = tk.Frame(d, bg=T.BG)
        kp.pack(fill="x")
        for i in range(3):
            kp.columnconfigure(i, weight=1, uniform="kp")
        defs = [("n", "Dans le système", "users", T.PRIMARY, "Nombre de clients présents à cet instant."),
                ("L", "L moyen", "layers", T.BLUE, "Nombre moyen de clients dans le système depuis le début."),
                ("W", "W moyen", "clock", T.GREEN, "Temps moyen passé dans le système."),
                ("util", "Occupation", "gauge", T.ORANGE, "Fraction du temps où les serveurs sont occupés."),
                ("refus", "Refusés", "ban", T.RED, "Part des arrivées refusées (système plein ou refus aléatoire)."),
                ("servis", "Servis", "check", "#5AC8FA", "Nombre de clients ayant terminé leur service.")]
        for i, (k, lib, ic, col, tip) in enumerate(defs):
            cd = Card(kp, pad=8, radius=14)
            cd.grid(row=i // 3, column=i % 3, sticky="nsew", padx=3, pady=3)
            self.sv[k] = tk.StringVar(value="—")
            tk.Label(cd.body, textvariable=self.sv[k], font=fnt(13, "bold"), fg=col, bg=T.CARD).pack(anchor="w")
            tk.Label(cd.body, text=lib, font=fnt(8), fg=T.MUTED, bg=T.CARD).pack(anchor="w")
            tip_all(cd, tip)
        cg = Card(d, pad=10, radius=16, expand=True)
        cg.pack(fill="both", expand=True, pady=(6, 0))
        tk.Label(cg.body, text="Clients dans le système au fil du temps", font=fnt(9, "bold"), fg=T.TEXT,
                 bg=T.CARD).pack(anchor="w")
        self.ctraj = tk.Canvas(cg.body, height=200, bg=T.CARD, highlightthickness=0)
        self.ctraj.pack(fill="both", expand=True)
        self.ctraj.bind("<Configure>", lambda e: self.maj_traj())
        TIP.attach(self.ctraj, "Courbe n(t) : nombre de clients présents. Lignes pointillées : c (serveurs) et K (capacité).")

        j = Card(dj, pad=12, radius=18, expand=True)
        j.pack(fill="both", expand=True)
        jb = j.body
        tete = tk.Frame(jb, bg=T.CARD)
        tete.pack(fill="x")
        ic = tk.Canvas(tete, width=24, height=24, bg=T.CARD, highlightthickness=0)
        ic.pack(side="left")
        draw_icon(ic, "list", 12, 12, 20, T.PRIMARY)
        tk.Label(tete, text="Journal", font=fnt(11, "bold"), fg=T.TEXT, bg=T.CARD).pack(side="left", padx=6)
        IconButton(tete, "trash", self.effacer_journal, "Effacer le journal", size=34, kind="ghost").pack(side="right")
        IconButton(tete, "save", self.enregistrer_journal, "Enregistrer le journal (Ctrl+S)", size=34,
                   kind="ghost").pack(side="right")
        self.verbose_var = tk.BooleanVar(value=True)
        cel = tk.Frame(tete, bg=T.CARD)
        cel.pack(side="right", padx=8)
        Toggle(cel, self.verbose_var, "Verbose : écrit chaque événement dans le journal.").pack(side="left")
        tk.Label(cel, text="Verbose", font=fnt(8), fg=T.MUTED, bg=T.CARD).pack(side="left", padx=3)
        corps = tk.Frame(jb, bg=T.CARD)
        corps.pack(fill="both", expand=True, pady=(6, 0))
        self.log = tk.Text(corps, width=38, height=8, wrap="word", state="disabled", font=(MONO, 9), bg="#F5F5F7",
                           fg=T.TEXT, relief="flat", bd=0, padx=8, pady=6, highlightthickness=1,
                           highlightbackground=T.BORDER, highlightcolor=T.BORDER)
        sb = tk.Scrollbar(corps, orient="vertical", command=self.log.yview, width=10, relief="flat",
                          troughcolor=T.CARD, bd=0)
        self.log.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y", padx=(4, 0))
        self.log.pack(side="left", fill="both", expand=True)
        gras = (MONO, 9, "bold")
        self.log.tag_configure("debut", foreground=T.PRIMARY_D, font=gras, spacing1=4)
        self.log.tag_configure("arr", foreground="#2563EB")
        self.log.tag_configure("start", foreground=T.PRIMARY_D)
        self.log.tag_configure("dep", foreground="#1E5FBF")
        self.log.tag_configure("block", foreground="#DC2626", font=gras)
        self.log.tag_configure("balk", foreground="#D97706")
        self.log.tag_configure("renege", foreground="#D97706", font=gras)
        TIP.attach(self.log, "Journal détaillé : arrivées, débuts de service, départs, refus et abandons.")


    @staticmethod
    def tu_par_s(v):
        return 0.2 * 10 ** (v / 100 * 3.4)

    def tip_scene(self, e):
        g = self.geo
        if not g:
            return None
        if e.x < g["x_q0"] - 30:
            return "Source de clients : infinie, ou population finie si N est fixé."
        if e.x < g["x_s0"] - 10:
            return ("File d'attente : les clients patientent ici dans l'ordre de la discipline choisie.\n"
                    "Leur visage se renfrogne quand l'attente s'allonge.")
        if e.x < g["x_s1"] + 10:
            return "Serveurs : l'anneau autour du client montre l'avancement de son service."
        return "Sortie : clients servis. Rouge = refusés (système plein), orange = abandons / refus aléatoires."

    # --- journal
    def ecrire(self, texte, tag=""):
        self.log.configure(state="normal")
        self.log.insert("end", texte + "\n", tag)
        n = int(self.log.index("end-1c").split(".")[0])
        if n > self.MAX_LIGNES:
            self.log.delete("1.0", f"{n - self.MAX_LIGNES + 400}.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def effacer_journal(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def enregistrer_journal(self):
        contenu = self.log.get("1.0", "end-1c")
        if not contenu.strip():
            messagebox.showinfo("Journal", "Le journal est vide.")
            return
        chemin = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Texte", "*.txt")],
                                              initialfile="journal_file_attente.txt")
        if chemin:
            with open(chemin, "w", encoding="utf-8") as f:
                f.write(contenu + "\n")
            self.toast("Journal enregistré")

    def decrire_evt(self, ev):
        t, kind, cid, info = ev
        pre = f"[t={t:8.2f}] "
        K = self.p["K"]
        if kind == "arr":
            n, pos = info
            if pos == 0:
                return pre + f"Client #{cid} arrive ({n} présent(s)) → serveur libre : service immédiat", "arr"
            return pre + f"Client #{cid} arrive ({n} présent(s)) → rejoint la file, position {pos}", "arr"
        if kind == "start":
            slot, w, svc = info
            srv = "un serveur" if self.p["c"] == INF else f"le serveur {slot + 1}"
            return pre + f"Client #{cid} commence sur {srv} après {w:.2f} d'attente (service prévu {svc:.2f})", "start"
        if kind == "dep":
            return pre + f"Client #{cid} termine et part (temps total {info[1]:.2f})", "dep"
        if kind == "block":
            return pre + f"Client #{cid} REFUSÉ : système plein ({info}/{int(K)})", "block"
        if kind == "balk":
            return pre + f"Client #{cid} renonce à entrer (refus aléatoire)", "balk"
        return pre + f"Client #{cid} perd patience après {info:.2f} et quitte la file", "renege"

    # --- cycle de vie de la simulation
    def nouvelle_sim_btn(self):
        self.nouvelle_sim()
        self.status.set("Simulation remise à zéro — appuyez sur Lecture")

    def nouvelle_sim(self):
        self.running = False
        self.ff = False
        self.sim = None
        self.clock = 0.0
        self.effets, self.mouv, self.pos = [], {}, {}
        self.cum = dict(arr=0, dep=0, block=0, balk=0, renege=0)
        if hasattr(self, "fab"):
            self.maj_fab()
            self.dessiner_scene()
            self.maj_live(True)
            self.maj_traj()
            self.status.set("Prêt — appuyez sur Lecture")

    def creer_sim(self):
        self.sim = Sim(self.p)
        self.clock = 0.0
        self.effets, self.mouv, self.pos = [], {}, {}
        self.cum = dict(arr=0, dep=0, block=0, balk=0, renege=0)
        self.ech_humeur = max(2.0 / self.p["mu"], self.p["patience"] or 0.0, 1.0)
        if self.verbose_var.get():
            self.ecrire("", "")
            self.ecrire(f"=== {kendall(self.p)} — graine {self.sim.seed} — horizon {self.p['horizon']:g} ===", "debut")

    def maj_fab(self):
        if self.ff:
            self.fab.reconfigurer("play", "Calcul…", "Calcul en cours")
        elif self.running:
            self.fab.reconfigurer("pause", "Pause", "Mettre la simulation en pause (Espace)")
        elif self.sim and not self.sim.finished():
            self.fab.reconfigurer("play", "Reprendre", "Reprendre la simulation (Espace)")
        else:
            self.fab.reconfigurer("play", "Lecture", "Lancer la simulation (Espace)")

    def fab_action(self):
        if self.ff:
            return
        if self.running:
            self.running = False
        else:
            if self.sim is None or self.fini():
                self.creer_sim()
            self.running = True
            self.last = time.perf_counter()
        self.maj_fab()

    def fini(self):
        return self.sim is not None and self.clock >= self.p["horizon"] - 1e-9

    def pas_sim(self):
        if self.ff:
            return
        self.running = False
        if self.sim is None or self.fini():
            self.creer_sim()
        ev = self.sim.step()
        if ev is None:
            self.finir()
            return
        self.clock = self.sim.t
        self.traiter(ev, time.perf_counter())
        self.maj_fab()
        self.status.set(f"Événement suivant — t = {self.clock:.2f}")
        self.dessiner_scene()
        self.maj_live(True)
        self.maj_traj()

    def fin_rapide(self):
        if self.ff:
            return
        self.running = False
        if self.sim is None or self.fini():
            self.creer_sim()
        self.ff = True
        self.maj_fab()
        self.status.set("Calcul rapide en cours…")

        def morceau():
            if not self.ff:
                return
            for _ in range(4000):
                if self.sim.step() is None:
                    self.ff = False
                    self.clock = self.p["horizon"]
                    self.finir()
                    return
            self.clock = self.sim.t
            self.maj_live(True)
            self.dessiner_scene()
            self.root.after(1, morceau)
        self.root.after(1, morceau)

    def finir(self):
        s = self.sim
        self.running = False
        self.ff = False
        s.close()
        self.clock = self.p["horizon"]
        m = s.metrics()
        self.res = dict(params=dict(self.p), th=theorie(self.p), agg={k: (v, 0.0) for k, v in m.items()}, reps=1,
                        pn=s.pn(), waits=list(s.waits), source="Simulation animée (1 réplication)")
        if self.verbose_var.get():
            self.ecrire(f"=== Fin à t = {self.p['horizon']:g} : L = {m['L']:.3f}, W = {m['W']:.3f}, "
                        f"utilisation = {m['util'] * 100:.1f} %, refus = {m['p_block'] * 100:.1f} % ===", "debut")
        self.maj_fab()
        self.dessiner_scene()
        self.maj_live(True)
        self.maj_traj()
        self.status.set("Simulation terminée — consultez l'écran Résultats")
        self.toast("Simulation terminée : comparez avec la théorie dans Résultats")

    # --- boucle d'animation
    def boucle(self):
        now = time.perf_counter()
        dt = min(now - self.last, 0.1)
        self.last = now
        if self.running and self.sim is not None:
            hz = self.p["horizon"]
            self.clock = min(self.clock + dt * self.tu_par_s(self.vit.slider.get()), hz)
            evs = self.sim.advance_to(self.clock)
            if evs:
                self.traiter(evs, now)
            self.sim._integ(self.clock)
            self.status.set(f"Simulation en cours — t = {self.clock:.1f} / {hz:g}")
            if self.clock >= hz - 1e-9:
                self.finir()
        self.effets = [e for e in self.effets if now - e["born"] < e["dur"]]
        self.mouv = {k: v for k, v in self.mouv.items() if now - v[2] < v[3] + 0.5}
        if self.page == "sim" and (self.running or self.effets or self.mouv):
            self.dessiner_scene()
        elif self.page == "suivi" and self.running and now - self._live > 0.15:
            self.maj_live()
            self.maj_traj()
            self._live = now
        # A 60 Hz cadence makes movement and service progress feel continuous.
        self.root.after(16, self.boucle)

    def position_animee(self, cid, cible, now):
        """Return a smoothly eased, slightly lifted position for a moving client."""
        mouvement = self.mouv.get(cid)
        if not mouvement:
            return cible
        x0, y0, debut, duree = mouvement
        u = min(1.0, max(0.0, (now - debut) / max(duree, 1e-6)))
        if u >= 1.0:
            self.mouv.pop(cid, None)
            return cible
        # Smoothstep avoids the abrupt starts and stops of a linear slide.
        lisse = u * u * (3.0 - 2.0 * u)
        x1, y1 = cible
        arc = math.sin(math.pi * u) * min(7.0, abs(x1 - x0) * .035)
        return x0 + (x1 - x0) * lisse, y0 + (y1 - y0) * lisse - arc

    def traiter(self, evs, now):
        g = self.geo or dict(door=(100, 100), exit=(500, 100), src=(60, 100))
        verb = self.verbose_var.get()
        manque = max(0, len(evs) - 60) if verb else 0
        for i, ev in enumerate(evs):
            t, kind, cid, info = ev
            if kind == "arr":
                self.cum["arr"] += 1
                self.mouv[cid] = (g["door"][0], g["door"][1], now, 0.35)
            elif kind == "start":
                x, y = self.pos.get(cid, g["door"])
                self.mouv[cid] = (x, y, now, 0.3)
            else:
                if len(self.effets) < 40:
                    p0 = self.pos.get(cid, g["src"] if kind in ("block", "balk") else g["exit"])
                    self.effets.append(dict(kind=kind, cid=cid, born=now, dur=0.9 if kind != "dep" else 0.8, p0=p0))
                self.cum[kind] += 1
            if verb and i < 60:
                txt, tag = self.decrire_evt(ev)
                self.ecrire(txt, tag)
        if manque:
            self.ecrire(f"… {manque} événement(s) non affichés (vitesse élevée)", "debut")

    # --- dessin de la scène
    def client(self, c, x, y, r, cid, humeur, ring=None):
        rr, gg, bb = colorsys.hsv_to_rgb((cid * 0.61803) % 1.0, .42, .97)
        corps = f"#{int(rr * 255):02x}{int(gg * 255):02x}{int(bb * 255):02x}"
        if humeur < .5:
            mood = mix(T.GREEN, T.ORANGE, humeur * 2)
        else:
            mood = mix(T.ORANGE, T.RED, (humeur - .5) * 2)
        ring = ring or mood
        lw = 2.5 if r >= 9 else 1.5
        sombre = "#1F2937"
        contour = "#111111"
        f = self.forme
        # A soft ground shadow gives every client a little depth without
        # relying on platform-specific transparency support in Tkinter.
        ombre_y = y + max(2, r * .62)
        c.create_oval(x - r * .68, ombre_y - max(1.5, r * .18),
                      x + r * .68, ombre_y + max(1.5, r * .18),
                      fill=mix(T.TEXT, T.CARD, .82), outline="")
        if f == "disque":
            c.create_oval(x - r, y - r, x + r, y + r, fill=corps, outline=contour, width=lw)
            if r >= 7:
                c.create_oval(x - r * .58, y - r * .66, x - r * .08, y - r * .28,
                              fill=mix(corps, "#FFFFFF", .58), outline="")
        elif f == "carre":
            # Network packet: squared body, folded corner and data lanes.
            s = r * .9
            pli = max(2, r * .32)
            c.create_polygon(x - s, y - s, x + s - pli, y - s, x + s, y - s + pli,
                             x + s, y + s, x - s, y + s, fill=corps, outline=contour,
                             width=lw, joinstyle="round")
            c.create_line(x + s - pli, y - s, x + s - pli, y - s + pli, x + s, y - s + pli,
                          fill=contour, width=max(1, lw * .7), joinstyle="round")
            if r >= 8:
                for lane in (-.25, .18):
                    c.create_line(x - s * .48, y + r * lane, x + s * .38, y + r * lane,
                                  fill=mix(corps, "#FFFFFF", .68), width=max(1, r * .12),
                                  capstyle="round")
        elif f == "voiture":
            w = r
            # Layered body, cabin, bumpers, lamps and wheel hubs give the
            # car a clearer silhouette at both small and large scales.
            c.create_polygon(x - w * .55, y - r * .02, x - w * .34, y - r * .72,
                             x + w * .27, y - r * .72, x + w * .63, y - r * .02,
                             fill=mix(corps, "#FFFFFF", .12), outline=contour, width=lw,
                             joinstyle="round")
            rrect(c, x - w, y - r * .12, x + w, y + r * .48, r * .22,
                  fill=corps, outline=contour, width=lw)
            if r >= 8:
                c.create_polygon(x - w * .39, y - r * .17, x - w * .24, y - r * .55,
                                 x + w * .19, y - r * .55, x + w * .41, y - r * .17,
                                 fill="#DCEBFF", outline=contour, width=max(1, lw * .55),
                                 joinstyle="round")
                c.create_line(x - w * .02, y - r * .54, x - w * .02, y - r * .16,
                              fill=mix(T.BLUE, "#FFFFFF", .25), width=max(1, r * .07))
                for lx in (x - w * .83, x + w * .66):
                    c.create_oval(lx, y - r * .03, lx + w * .16, y + r * .16,
                                  fill="#FDE68A" if lx > x else "#FCA5A5", outline=contour,
                                  width=max(1, lw * .4))
                c.create_line(x - w * .82, y + r * .32, x + w * .82, y + r * .32,
                              fill=mix(corps, "#000000", .22), width=max(1, r * .08))
            for wx in (x - w * .52, x + w * .52):
                c.create_oval(wx - r * .27, y + r * .25, wx + r * .27, y + r * .79,
                              fill=sombre, outline=contour, width=max(1, lw * .55))
                if r >= 10:
                    c.create_oval(wx - r * .11, y + r * .42, wx + r * .11, y + r * .64,
                                  fill="#D1D5DB", outline=contour, width=1)
        elif f == "bonhomme":
            lg = max(1.5, r * .16)
            femme = cid % 5 == 0
            if femme:
                # Roughly one client in five has a feminine silhouette.
                # Draw hair first so the face and dress sit cleanly above it.
                c.create_oval(x - r * .40, y - r, x + r * .40, y - r * .17,
                              fill=mix(corps, "#302719", .48), outline=contour, width=lw)
                c.create_polygon(x - r * .23, y - r * .26, x + r * .23, y - r * .26,
                                 x + r * .43, y + r * .40, x - r * .43, y + r * .40,
                                 fill=corps, outline=contour, width=lw, joinstyle="round")
                c.create_line(x - r * .48, y - r * .07, x - r * .22, y - r * .20,
                              x + r * .22, y - r * .20, x + r * .48, y - r * .07,
                              fill=contour, width=lg, capstyle="round", joinstyle="round")
                c.create_line(x - r * .20, y + r * .40, x - r * .18, y + r * .84,
                              x + r * .18, y + r * .84, x + r * .20, y + r * .40,
                              fill=contour, width=lg, capstyle="round", joinstyle="round")
                c.create_oval(x - r * .27, y - r * .88, x + r * .27, y - r * .36,
                              fill=corps, outline=contour, width=lw)
                if r >= 10:
                    c.create_line(x - r * .35, y - r * .58, x - r * .30, y - r * .24,
                                  fill=contour, width=max(1, lw * .65))
                    c.create_line(x + r * .35, y - r * .58, x + r * .30, y - r * .24,
                                  fill=contour, width=max(1, lw * .65))
            else:
                c.create_line(x, y - r * .3, x, y + r * .25, fill=contour, width=lg, capstyle="round")
                c.create_line(x - r * .5, y - r * .05, x, y - r * .22, x + r * .5, y - r * .05, fill=contour, width=lg,
                              capstyle="round", joinstyle="round")
                c.create_line(x - r * .42, y + r * .85, x, y + r * .25, x + r * .42, y + r * .85, fill=contour, width=lg,
                              capstyle="round", joinstyle="round")
                c.create_oval(x - r * .34, y - r * .95, x + r * .34, y - r * .3, fill=corps,
                              outline=contour, width=lw)
        elif f == "etoile":
            pp = []
            for k in range(10):
                a_ = -math.pi / 2 + k * math.pi / 5
                rad = r if k % 2 == 0 else r * .45
                pp += [x + rad * math.cos(a_), y + rad * math.sin(a_)]
            c.create_polygon(*pp, fill=corps, outline=contour, width=lw, joinstyle="round")
        else:
            c.create_oval(x - r, y - r, x + r, y + r, fill=corps, outline=contour, width=lw)
            if r < 8:
                return
            c.create_oval(x - r * .70, y - r * .76, x - r * .38, y - r * .52,
                          fill=mix(corps, "#FFFFFF", .48), outline="")
            e = r * .32
            er = max(1.2, r * .11)
            for ex in (x - e, x + e):
                c.create_oval(ex - er, y - r * .22 - er, ex + er, y - r * .22 + er, fill=sombre, outline="")
            if humeur < .35:
                c.create_arc(x - r * .42, y - r * .05, x + r * .42, y + r * .55, start=205, extent=130, style="arc",
                             outline=sombre, width=max(1, r * .1))
            elif humeur < .7:
                c.create_line(x - r * .3, y + r * .3, x + r * .3, y + r * .3, fill=sombre, width=max(1, r * .1))
            else:
                c.create_arc(x - r * .42, y + r * .22, x + r * .42, y + r * .8, start=25, extent=130, style="arc",
                             outline=sombre, width=max(1, r * .1))

    def humeur(self, cl):
        return min(1.0, max(0.0, (self.clock - cl.t_arr) / self.ech_humeur)) if cl.t_start is None else \
            min(1.0, (cl.t_start - cl.t_arr) / self.ech_humeur)

    def dessiner_scene(self):
        if not hasattr(self, "scene"):
            return
        c = self.scene
        c.delete("all")
        w, h = dims(c)
        rrect(c, 2, 2, w - 2, h - 2, 28, fill=T.CARD, outline="")
        p, sim = self.p, self.sim
        now = time.perf_counter()
        cs = p["c"]
        K = p["K"]
        x_src, x_q0, x_s0, x_s1, x_exit = 62, 128, w * 0.60, w - 124, w - 58
        y_mid = h / 2 + 6
        x_q1 = x_s0 - 26
        self.geo = dict(door=(x_q0 - 16, y_mid), exit=(x_exit, y_mid), src=(x_src + 26, y_mid), x_q0=x_q0, x_s0=x_s0,
                        x_s1=x_s1)
        # Dotted boundary makes the system definition explicit: queue plus servers.
        bx0, by0, bx1, by1 = x_q0 - 10, 54, x_s1 + 8, h - 50
        c.create_rectangle(bx0, by0, bx1, by1, outline="#111111", width=1.5, dash=(5, 4))
        # Small white backing keeps the system title legible over the dotted edge.
        sys_x = (bx0 + bx1) / 2
        rrect(c, sys_x - 43, by0 - 8, sys_x + 43, by0 + 10, 7, fill=T.CARD, outline="")
        c.create_text(sys_x, by0 + 1, text="SYSTÈME", font=fnt(9, "bold"), fill=T.TEXT)
        c.create_text((x_q0 + x_q1) / 2, 66, text="FILE D'ATTENTE", font=fnt(8, "bold"), fill=T.MUTED)
        c.create_text((x_s0 + x_s1) / 2, 66, text="SERVEUR", font=fnt(8, "bold"), fill=T.MUTED)
        c.create_text(x_src, 24, text="SOURCE", font=fnt(9, "bold"), fill=T.MUTED)
        c.create_text(x_exit, 24, text="SORTIE", font=fnt(9, "bold"), fill=T.MUTED)
        # source
        c.create_oval(x_src - 26, y_mid - 26, x_src + 26, y_mid + 26, fill=T.PRIMARY_L, outline="")
        draw_icon(c, "users", x_src, y_mid, 28, T.PRIMARY)
        c.create_text(x_src, y_mid + 42, text="∞" if p["N"] == INF else f"N = {int(p['N'])}", font=fnt(9, "bold"),
                      fill=T.PRIMARY_D)
        c.create_line(x_src + 30, y_mid, x_q0 - 8, y_mid, fill=T.BORDER, width=3, arrow="last")
        # file
        sz = int(max(30, min(58, h / 9)))
        gap = 0
        self.rcl = sz / 2 - 2
        per_row = max(3, int((x_q1 - x_q0) // (sz + gap)))
        rows_max = max(1, min(6, int((h - 150) // (sz + gap + 4))))
        vis = per_row * rows_max
        file_q = list(sim.queue)[:vis] if sim else []
        nq = len(sim.queue) if sim else 0
        ns = max(0, K - (cs if cs != INF else 0)) if K != INF else INF
        n_show = min(vis, ns) if ns != INF else min(vis, max(nq + 2, 10))
        rows_used = max(1, math.ceil(n_show / per_row)) if n_show else 1

        def slot_xy(i):
            row, col = i // per_row, i % per_row
            return (x_q1 - (col + .5) * (sz + gap),
                    y_mid - rows_used * (sz + gap) / 2 + (row + .5) * (sz + gap))
        for i in range(int(n_show)):
            x, y = slot_xy(i)
            c.create_rectangle(x - sz / 2, y - sz / 2, x + sz / 2, y + sz / 2,
                               fill="#F5F5F7", outline="#111111", width=1.3)
        if ns == 0:
            c.create_text((x_q0 + x_q1) / 2, y_mid, text="aucune salle d'attente (K = c)", font=fnt(9), fill=T.MUTED)
        newpos = {}
        for i, cl in enumerate(file_q):
            x, y = self.position_animee(cl.id, slot_xy(i), now)
            newpos[cl.id] = (x, y)
            self.client(c, x, y, sz / 2 - 2, cl.id, self.humeur(cl))
        if nq > vis:
            c.create_text(x_q0 + 24, y_mid, text=f"+{nq - vis}", font=fnt(14, "bold"), fill=T.RED)
        if K != INF:
            c.create_text((x_q0 + x_q1) / 2, y_mid + rows_used * (sz + gap) / 2 + 18, text=f"capacité totale K = {int(K)}",
                          font=fnt(8), fill=T.MUTED)
        # serveurs
        by_slot = {cl.slot: cl for cl in sim.busy.values()} if sim else {}
        c.create_line(x_q1 + 4, y_mid, x_s0 + 4, y_mid, fill=T.BORDER, width=3, arrow="last")
        if cs == INF:
            pw, ph = x_s1 - x_s0, min(h - 120, 180)
            c.create_rectangle(x_s0, y_mid - ph / 2, x_s1, y_mid + ph / 2,
                               fill="#F5F5F7", outline="#111111", width=1.5)
            busy = sorted(sim.busy.values(), key=lambda k: k.slot) if sim else []
            r = 10
            per = max(1, int((pw - 16) // (2 * r + 4)))
            for i, cl in enumerate(busy[:per * 6]):
                x = x_s0 + 14 + r + (i % per) * (2 * r + 4)
                y = y_mid - ph / 2 + 22 + r + (i // per) * (2 * r + 4)
                newpos[cl.id] = (x, y)
                self.client(c, x, y, r, cl.id, self.humeur(cl))
            c.create_text((x_s0 + x_s1) / 2, y_mid + ph / 2 - 12, text=f"{len(busy)} en service (∞ serveurs)",
                          font=fnt(8, "bold"), fill=T.MUTED)
        else:
            nc = int(cs)
            mh = min(124, max(22, (h - 150) / nc))
            mw = min(x_s1 - x_s0, 230)
            mx0 = x_s0 + (x_s1 - x_s0 - mw) / 2
            top = y_mid - (nc * mh) / 2
            for s in range(nc):
                y0 = top + s * mh
                cl = by_slot.get(s)
                c.create_rectangle(mx0, y0, mx0 + mw, y0 + mh,
                                   fill=mix(T.PRIMARY, "#FFFFFF", .92) if cl else "#F5F5F7",
                                   outline="#111111", width=1.3)
                cx, cy = mx0 + mh / 2 + 2, y0 + mh / 2
                r = min(mh / 2 - 7, 30)
                c.create_text(mx0 + mw - 10, y0 + 12, text=f"S{s + 1}", anchor="e", font=fnt(8, "bold"), fill=T.MUTED)
                if cl:
                    prog = min(1.0, max(0.0, (self.clock - cl.t_start) / cl.svc))
                    c.create_oval(cx - r - 5, cy - r - 5, cx + r + 5, cy + r + 5, outline="#DDE3EE", width=4)
                    if prog > 0:
                        c.create_arc(cx - r - 5, cy - r - 5, cx + r + 5, cy + r + 5, start=90, extent=-359.9 * prog,
                                     style="arc", outline=T.PRIMARY, width=4)
                    x, y = self.position_animee(cl.id, (cx, cy), now)
                    newpos[cl.id] = (x, y)
                    self.client(c, x, y, r, cl.id, self.humeur(cl))
                    c.create_text(mx0 + mw - 10, y0 + mh - 12, text=f"{int(prog * 100)} %", anchor="e", font=fnt(8),
                                  fill=T.PRIMARY_D)
                else:
                    c.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#111111", dash=(3, 3), width=1.5)
                    c.create_text(mx0 + mw - 10, y0 + mh - 12, text="libre", anchor="e", font=fnt(8), fill=T.DISABLED)
        self.pos = newpos
        # sortie
        c.create_line(x_s1 + 8, y_mid, x_exit - 22, y_mid, fill=T.BORDER, width=3, arrow="last")
        c.create_oval(x_exit - 22, y_mid - 22, x_exit + 22, y_mid + 22, fill=mix(T.GREEN, "#FFFFFF", .85), outline="")
        draw_icon(c, "door", x_exit, y_mid, 24, T.PRIMARY_D)
        # effets
        for ef in self.effets:
            u = min(1.0, (now - ef["born"]) / ef["dur"])
            x0, y0 = ef["p0"]
            k = ef["kind"]
            if k == "dep":
                x, y, ring = x0 + (x_exit - x0) * u, y0 + (y_mid - y0) * u, T.GREEN
            elif k == "block":
                dx, dy = self.geo["door"]
                if u < .5:
                    x, y = x0 + (dx - x0) * u * 2, y0 + (dy - y0) * u * 2
                else:
                    v = (u - .5) * 2
                    x, y = dx - 18 * v, dy - 70 * v
                ring = T.RED
            elif k == "balk":
                dx = x0 + (self.geo["door"][0] - x0) * .45
                x, y = (x0 + (dx - x0) * u * 2, y0) if u < .5 else (dx - 10 * (u - .5) * 2, y0 - 70 * (u - .5) * 2)
                ring = T.ORANGE
            else:
                x, y, ring = x0, y0 - 80 * u, T.ORANGE
            self.client(c, x, y, self.rcl * (1 - .35 * u), ef["cid"], .1 if k == "dep" else .9, ring)
            if k in ("block", "balk", "renege") and u > .3:
                draw_icon(c, "close", x + 12, y - 14, 12, ring)
        # compteurs
        x = 18
        for lib, val, col in (("Servis", self.cum["dep"], T.GREEN), ("Refusés", self.cum["block"], T.RED),
                              ("Abandons", self.cum["renege"], T.ORANGE), ("Renoncent", self.cum["balk"], T.MUTED)):
            c.create_oval(x, h - 28, x + 10, h - 18, fill=col, outline="")
            t = c.create_text(x + 15, h - 23, text=f"{lib} {val}", anchor="w", font=fnt(9, "bold"), fill=T.TEXT)
            x = c.bbox(t)[2] + 18
        c.create_text(w - 18, h - 23, anchor="e", font=fnt(9, "bold"), fill=T.PRIMARY_D,
                      text=f"t = {self.clock:.2f}  /  {p['horizon']:g}")
        if sim is None:
            c.create_text(w / 2, h - 52, text="Appuyez sur Lecture : les clients vont arriver", font=fnt(10), fill=T.MUTED)

    def maj_live(self, force=False):
        s = self.sim
        if not hasattr(self, "sv") or "n" not in self.sv:
            return
        if s is None:
            for k in ("n", "L", "W", "util", "refus", "servis"):
                self.sv[k].set("—")
            return
        m = s.metrics(self.clock)
        self.sv["n"].set(str(s.n()))
        self.sv["L"].set(f"{m['L']:.2f}")
        self.sv["W"].set(f"{m['W']:.2f}" if m["W"] == m["W"] else "—")
        self.sv["util"].set(f"{m['util'] * 100:.0f} %" if s.c != INF else f"{m['busy']:.2f}")
        tot = (m["p_block"] + m["p_balk"]) if m["arrivals"] else 0.0
        self.sv["refus"].set(f"{tot * 100:.1f} %" if m["arrivals"] else "—")
        self.sv["servis"].set(str(self.cum["dep"]))

    def maj_traj(self):
        if hasattr(self, "ctraj"):
            graphique_traj(self.ctraj, self.sim, self.clock, 200.0)

    # ------------------------------------------------------------------
    #  Page Résultats
    # ------------------------------------------------------------------
    def construire_res(self):
        p = self.pages["res"] = tk.Frame(self.conteneur, bg=T.BG)
        top = Card(p, pad=12, radius=18)
        top.pack(fill="x", padx=14, pady=(14, 6))
        t = tk.Frame(top.body, bg=T.CARD)
        t.pack(fill="x")
        t.columnconfigure(2, weight=1)
        self.btn_analyse = PillButton(t, "Lancer l'analyse", self.lancer_analyse, icon="flask",
                                      tip="Simule le modèle courant sur plusieurs réplications indépendantes "
                                          "pour obtenir des intervalles de confiance.")
        self.btn_analyse.grid(row=0, column=0, padx=(0, 16))
        self.reg_reps = Reglage(self, t, "Réplications", 1, 50, 10, 1, lambda v: f"{int(v)}", lambda v: None,
                                "Nombre de simulations indépendantes (plus il y en a, plus l'IC est étroit).", "layers")
        self.reg_reps.frame.grid(row=0, column=1, sticky="ew", padx=(0, 16))
        t.columnconfigure(1, weight=0, minsize=220)
        self.prog = tk.Canvas(t, height=12, bg=T.CARD, highlightthickness=0)
        self.prog.grid(row=0, column=2, sticky="ew", padx=(0, 12))
        TIP.attach(self.prog, "Avancement de l'analyse.")
        IconButton(t, "download", self.exporter_csv, "Exporter les résultats en CSV (Ctrl+E)", size=40).grid(row=0, column=3)
        self.lbl_src = tk.Label(top.body, text="Aucun résultat pour l'instant.", font=fnt(9), fg=T.MUTED, bg=T.CARD,
                                anchor="w", justify="left")
        self.lbl_src.pack(fill="x", pady=(6, 0))

        ct = Card(p, pad=12, radius=18)
        ct.pack(fill="x", padx=14, pady=6)
        CompareTable.ROW = 33
        self.table = CompareTable(ct.body, 8)
        self.table.pack(fill="x")

        bas = tk.Frame(p, bg=T.BG)
        bas.pack(fill="both", expand=True, padx=9, pady=(0, 12))
        bas.columnconfigure(0, weight=1, uniform="g")
        bas.columnconfigure(1, weight=1, uniform="g")
        bas.rowconfigure(0, weight=1)
        self.cpn = self.carte_graphe(bas, 0, "Distribution du nombre de clients")
        self.chist = self.carte_graphe(bas, 1, "Temps d'attente avant service")

    def carte_graphe(self, parent, col, titre, row=0):
        c = Card(parent, pad=12, expand=True)
        c.grid(row=row, column=col, sticky="nsew", padx=5)
        tk.Label(c.body, text=titre, font=fnt(10, "bold"), fg=T.TEXT, bg=T.CARD).pack(anchor="w")
        cv = tk.Canvas(c.body, bg=T.CARD, highlightthickness=0, height=140)
        cv.pack(fill="both", expand=True)
        cv.bind("<Configure>", lambda e: self.maj_resultats_graphes())
        TIP.attach(cv, lambda e, cv=cv: hit_text(cv, e), delay=120)
        return cv

    def progression(self, i, n):
        c = self.prog
        c.delete("all")
        w = max(c.winfo_width(), 50)
        rrect(c, 0, 2, w, 10, 4, fill="#E5EAF3", outline="")
        if i:
            rrect(c, 0, 2, max(8, w * i / n), 10, 4, fill=T.PRIMARY, outline="")

    def lancer_analyse(self):
        if self.occupe:
            self.toast("Un calcul est déjà en cours")
            return
        reps = int(self.reg_reps.slider.get())
        params = dict(self.p)
        self.btn_analyse.set_enabled(False)
        self.lbl_src.config(text=f"Analyse en cours : {reps} réplication(s) de {params['horizon']:g} u.t. …")

        def fin(etat):
            self.btn_analyse.set_enabled(True)
            ms, pns = etat["ms"], etat["pns"]
            n = max(len(x) for x in pns)
            moy = [sum((x[i] if i < len(x) else 0.0) for x in pns) / len(pns) for i in range(n)]
            self.res = dict(params=params, th=theorie(params), agg=agreger(ms), reps=len(ms), pn=moy,
                            waits=etat["waits"], source=f"Analyse : {len(ms)} réplications indépendantes, IC à 95 %")
            self.maj_resultats()
            self.toast("Analyse terminée")
        self.batch(params, reps, self.progression, fin)

    def batch(self, params, reps, progress, done):
        self.occupe = True
        seed0 = params.get("seed") or random.randrange(1, 10 ** 6)
        etat = dict(i=0, ms=[], pns=[], waits=[])

        def etape():
            try:
                m, pn, w = lancer(params, seed0 + 17 * etat["i"])
            except Exception as ex:
                self.occupe = False
                messagebox.showerror("Erreur de simulation", str(ex))
                return
            etat["ms"].append(m)
            etat["pns"].append(pn)
            if etat["i"] < 3:
                etat["waits"].extend(w[:20000])
            etat["i"] += 1
            progress(etat["i"], reps)
            if etat["i"] < reps:
                self.root.after(1, etape)
            else:
                self.occupe = False
                done(etat)
        self.root.after(1, etape)

    def maj_resultats(self):
        r = self.res
        if r is None:
            self.table.set_rows([])
            self.maj_resultats_graphes()
            return
        th, agg, p = r["th"], r["agg"], r["params"]
        pct = {"p_block", "p_renege", "util"}
        cinf = p["c"] == INF
        spec = [("L", "Clients dans le système (L)", "users", T.PRIMARY, "Nombre moyen de clients présents (file + service)."),
                ("Lq", "Clients en attente (Lq)", "layers", T.BLUE, "Nombre moyen de clients dans la file."),
                ("W", "Temps dans le système (W)", "clock", T.GREEN, "Temps moyen entre l'arrivée et le départ."),
                ("Wq", "Temps d'attente (Wq)", "hourglass", "#5AC8FA", "Temps moyen avant le début du service."),
                ("util", "Serveurs occupés" if cinf else "Utilisation des serveurs", "gauge", T.ORANGE,
                 "Fraction du temps où un serveur travaille (ou nombre moyen de serveurs occupés si c = ∞)."),
                ("lam_eff", "Débit accepté (λ_eff)", "bolt", "#8B5CF6", "Clients admis par unité de temps."),
                ("p_block", "Probabilité de blocage", "ban", T.RED, "Part des arrivées refusées car le système est plein."),
                ("p_renege", "Probabilité d'abandon", "close", T.ORANGE, "Part des clients admis qui perdent patience.")]
        rows = []
        for key, nom, ico, col, tip in spec:
            tv = th.get(key) if th else None
            sv, ic = agg[key]
            isp = key in pct and not (key == "util" and cinf)

            def f(v, ic=0.0):
                if v is None or v != v:
                    return "—"
                s = f"{v * 100:.2f} %" if isp else f"{v:.3f}"
                if ic > 0:
                    s += f" ± {ic * 100:.2f}" if isp else f" ± {ic:.3f}"
                return s
            ecart, verdict = "", None
            if tv is not None and sv == sv:
                d = sv - tv
                ecart = f"{d * 100:+.2f} pt" if isp else (f"{d / tv * 100:+.1f} %" if abs(tv) > 1e-9 else f"{d:+.3f}")
                tol = max(ic, 0.03 * abs(tv), 0.004)
                if abs(d) <= tol:
                    verdict = "ok" if th["nature"] == "exacte" else "approx"
                elif th["nature"] == "approchée" and abs(d) <= 0.2 * max(abs(tv), 1e-6):
                    verdict = "approx"
                else:
                    verdict = "ko"
            rows.append(dict(nom=nom, icon=ico, col=col, th=f(tv), sim=f(sv, ic), ecart=ecart, verdict=verdict,
                             tip=tip + ("\nThéorie : " + th["note"] if th and tv is not None else "")))
        self.table.set_rows(rows)
        note = th["note"] if th else ""
        self.lbl_src.config(text=f"{r['source']}  ·  Modèle {kendall(p)}  ·  {note}")
        self.maj_resultats_graphes()

    def maj_resultats_graphes(self):
        r = self.res
        if not hasattr(self, "chist"):
            return
        if r is None:
            graphique_pn(self.cpn, [], None)
            graphique_hist(self.chist, [], float("nan"))
            return
        graphique_pn(self.cpn, r["pn"], r["th"].get("pn") if r["th"] else None)
        graphique_hist(self.chist, r["waits"], r["agg"]["Wq"][0])

    def exporter_csv(self):
        r = self.res
        if r is None:
            messagebox.showinfo("Export CSV", "Lancez d'abord une simulation ou une analyse.")
            return
        chemin = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                              initialfile="resultats_file_attente.csv")
        if not chemin:
            return
        p = r["params"]
        with open(chemin, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["Modèle", kendall(p)])
            for k in ("lam", "mu", "c", "K", "N", "disc", "patience", "balk", "horizon", "warmup"):
                w.writerow([k, p[k]])
            w.writerow(["Réplications", r["reps"]])
            w.writerow([])
            w.writerow(["Indicateur", "Théorie", "Simulation", "IC95 (demi-largeur)"])
            th = r["th"] or {}
            for k, (m, ic) in r["agg"].items():
                w.writerow([k, th.get(k, ""), m, ic])
            w.writerow([])
            w.writerow(["n", "P(n) simulation", "P(n) théorie"])
            pt = th.get("pn") or []
            for i, v in enumerate(r["pn"]):
                w.writerow([i, v, pt[i] if i < len(pt) else ""])
        self.toast("Résultats exportés")

    # ------------------------------------------------------------------
    #  Page Études (balayage + défis)
    # ------------------------------------------------------------------
    def construire_etudes(self):
        p = self.pages["etudes"] = tk.Frame(self.conteneur, bg=T.BG)
        haut = tk.Frame(p, bg=T.BG)
        haut.pack(fill="x", padx=14, pady=(14, 8))
        self.tabs_etudes = ChipGroup(haut, [("bal", "Balayage de paramètre"), ("defis", "Défis")], "bal",
                                     self.vue_etudes, {"bal": "Fait varier un paramètre et trace l'indicateur choisi.",
                                                       "defis": "Cinq situations concrètes à optimiser, avec étoiles."})
        self.tabs_etudes.pack(side="left")
        self.lbl_etoiles = tk.Label(haut, text="", font=fnt(10, "bold"), fg="#B45309", bg=T.BG)
        self.lbl_etoiles.pack(side="right")
        self.zone_etudes = tk.Frame(p, bg=T.BG)
        self.zone_etudes.pack(fill="both", expand=True)

        # --- vue balayage
        v = self.vue_bal = tk.Frame(self.zone_etudes, bg=T.BG)
        v.columnconfigure(1, weight=1)
        v.rowconfigure(0, weight=1)
        ctrl = Card(v, pad=14, radius=18)
        ctrl.grid(row=0, column=0, sticky="ns", padx=(14, 7), pady=(0, 14))
        b = ctrl.body
        tk.Label(b, text="Paramètre à faire varier", font=fnt(9, "bold"), fg=T.TEXT, bg=T.CARD).pack(anchor="w")
        self.chips_bvar = ChipGroup(b, [("lam", "Charge λ"), ("c", "Serveurs c"), ("K", "Capacité K")], "lam",
                                    lambda k: setattr(self, "bal_var", k) or self.maj_balayage_vide(),
                                    {"lam": "Fait varier le taux d'arrivée λ de 10 % à 140 % de la capacité c·μ.",
                                     "c": "Fait varier le nombre de serveurs de 1 à 8.",
                                     "K": "Fait varier la capacité du système de c à 30."})
        self.chips_bvar.pack(fill="x", pady=(4, 10))
        tk.Label(b, text="Indicateur", font=fnt(9, "bold"), fg=T.TEXT, bg=T.CARD).pack(anchor="w")
        self.chips_bmet = ChipGroup(b, [("W", "W"), ("Wq", "Wq"), ("L", "L"), ("p_block", "P(blocage)"),
                                        ("util", "Utilisation")], "W",
                                    lambda k: setattr(self, "bal_met", k) or self.maj_balayage_vide(),
                                    {"W": "Temps moyen dans le système.", "Wq": "Temps d'attente moyen.",
                                     "L": "Nombre moyen de clients dans le système.",
                                     "p_block": "Probabilité qu'un client soit refusé.",
                                     "util": "Utilisation des serveurs."})
        self.chips_bmet.pack(fill="x", pady=(4, 10))
        self.reg_bal = Reglage(self, b, "Réplications / point", 2, 20, 5, 1, lambda v: f"{int(v)}", lambda v: None,
                               "Nombre de simulations par point du balayage.", "layers")
        self.reg_bal.frame.pack(fill="x")
        self.btn_bal = PillButton(b, "Lancer le balayage", self.lancer_balayage, icon="flask",
                                  tip="Lance le balayage avec le modèle courant (écran Modèle).")
        self.btn_bal.pack(pady=(12, 6))
        self.prog_bal = tk.Canvas(b, height=12, bg=T.CARD, highlightthickness=0)
        self.prog_bal.pack(fill="x")
        tk.Label(b, text="Les autres paramètres sont ceux de l'écran Modèle.\nCourbe pointillée = théorie (si elle existe).",
                 font=fnt(8), fg=T.MUTED, bg=T.CARD, justify="left").pack(anchor="w", pady=(10, 0))
        cg = Card(v, pad=12, radius=18, expand=True)
        cg.grid(row=0, column=1, sticky="nsew", padx=(7, 14), pady=(0, 14))
        self.titre_bal = tk.Label(cg.body, text="Balayage", font=fnt(11, "bold"), fg=T.TEXT, bg=T.CARD)
        self.titre_bal.pack(anchor="w")
        self.cxy = tk.Canvas(cg.body, bg=T.CARD, highlightthickness=0, height=300)
        self.cxy.pack(fill="both", expand=True)
        self.cxy.bind("<Configure>", lambda e: self.redessiner_balayage())
        TIP.attach(self.cxy, lambda e: hit_text(self.cxy, e), delay=120)
        self.bal_data = None

        # --- vue défis
        self.vue_def = ScrollFrame(self.zone_etudes)
        inn = self.vue_def.inner
        inn.columnconfigure(0, weight=1, uniform="d")
        inn.columnconfigure(1, weight=1, uniform="d")
        self.defi_ui = {}
        for i, d in enumerate(DEFIS):
            card = Card(inn, pad=14)
            card.grid(row=i // 2, column=i % 2, sticky="nsew", padx=(14 if i % 2 == 0 else 6, 6 if i % 2 == 0 else 14),
                      pady=(0, 12))
            self.construire_defi(card.body, d)
        self.vue_etudes("bal")

    def vue_etudes(self, cle):
        self.tabs_etudes.set(cle)
        self.vue_bal.pack_forget()
        self.vue_def.pack_forget()
        if cle == "bal":
            self.vue_bal.pack(fill="both", expand=True)
            self.root.after(40, self.redessiner_balayage)
        else:
            self.vue_def.pack(fill="both", expand=True)
        self.maj_etoiles()

    def maj_etoiles(self):
        tot = sum(self.etoiles.values())
        self.lbl_etoiles.config(text=f"★ {tot} / {3 * len(DEFIS)}")

    def maj_balayage_vide(self):
        if self.bal_data is None:
            self.redessiner_balayage()

    def lancer_balayage(self):
        if self.occupe:
            self.toast("Un calcul est déjà en cours")
            return
        var, met = self.bal_var, self.bal_met
        reps = int(self.reg_bal.slider.get())
        base = normaliser(dict(self.p))
        base["horizon"] = min(base["horizon"], 4000.0)
        base["warmup"] = base["warmup"] or base["horizon"] * 0.1
        c, mu = base["c"], base["mu"]
        if var == "lam":
            cmu = (c if c != INF else 4) * mu
            fmax = 1.4 if (base["K"] != INF or base["patience"] or base["N"] != INF) else 0.97
            xs = [round(cmu * (0.1 + (fmax - 0.1) * i / 13), 4) for i in range(14)]
            xl = "λ (arrivées par u.t.)"
        elif var == "c":
            xs = list(range(1, 9))
            xl = "serveurs c"
        else:
            cm = 1 if c == INF else int(c)
            xs = [k for k in (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30) if k >= cm]
            xl = "capacité K"
        noms = {"W": "W (temps dans le système)", "Wq": "Wq (attente)", "L": "L (clients présents)",
                "p_block": "P(blocage)", "util": "utilisation"}
        data = dict(xs=xs, sim=[None] * len(xs), ic=[0.0] * len(xs), th=[None] * len(xs), xl=xl, yl=noms[met], met=met)
        self.bal_data = data
        self.titre_bal.config(text=f"{noms[met]} en fonction de {xl}")
        self.btn_bal.set_enabled(False)
        idx = {"i": 0}

        def point():
            i = idx["i"]
            if i >= len(xs):
                self.btn_bal.set_enabled(True)
                self.toast("Balayage terminé")
                return
            pi = normaliser(dict(base, **{var: xs[i]}))
            th = theorie(pi)
            data["th"][i] = th.get(met) if th["nature"] else None

            def fin(etat):
                ag = agreger(etat["ms"])
                data["sim"][i], data["ic"][i] = ag[met]
                self.redessiner_balayage()
                idx["i"] += 1
                self.root.after(1, point)
            self.batch(pi, reps, lambda a, b: self.prog_bal_maj(i * reps + a, len(xs) * reps), fin)
        point()

    def prog_bal_maj(self, i, n):
        c = self.prog_bal
        c.delete("all")
        w = max(c.winfo_width(), 50)
        rrect(c, 0, 2, w, 10, 4, fill="#E5EAF3", outline="")
        if i:
            rrect(c, 0, 2, max(8, w * i / n), 10, 4, fill=T.PRIMARY, outline="")

    def redessiner_balayage(self):
        if not hasattr(self, "cxy"):
            return
        d = self.bal_data
        if d is None:
            graphique_xy(self.cxy, [], [], "", "")
            return
        series = [dict(label="Simulation (IC 95 %)", ys=d["sim"], ic=d["ic"], col=T.PRIMARY)]
        if any(v is not None for v in d["th"]):
            series.append(dict(label="Théorie", ys=d["th"], col=T.ORANGE, dash=True, pts=False))
        graphique_xy(self.cxy, d["xs"], series, d["xl"], d["yl"])

    # --- défis
    def construire_defi(self, b, d):
        t = tk.Frame(b, bg=T.CARD)
        t.pack(fill="x")
        ic = tk.Canvas(t, width=34, height=34, bg=T.CARD, highlightthickness=0)
        ic.pack(side="left")
        ic.create_oval(1, 1, 33, 33, fill=mix(T.ORANGE, "#FFFFFF", .8), outline="")
        draw_icon(ic, d["icon"], 17, 17, 19, "#B45309")
        tk.Label(t, text=d["titre"], font=fnt(12, "bold"), fg=T.TEXT, bg=T.CARD).pack(side="left", padx=8)
        etoiles = tk.Canvas(t, width=84, height=26, bg=T.CARD, highlightthickness=0)
        etoiles.pack(side="right")
        TIP.attach(etoiles, "Étoiles obtenues (3 paliers de performance).")
        tk.Label(b, text=d["histoire"], font=fnt(9), fg=T.TEXT, bg=T.CARD, justify="left", anchor="w",
                 wraplength=430).pack(fill="x", pady=(8, 4))
        tk.Label(b, text="Objectif : " + d["objectif"], font=fnt(9, "bold"), fg=T.PRIMARY_D, bg=T.CARD, justify="left",
                 anchor="w", wraplength=430).pack(fill="x", pady=(0, 4))
        tk.Label(b, text="Indice : " + d["indice"], font=fnt(8), fg=T.MUTED, bg=T.CARD, justify="left", anchor="w",
                 wraplength=430).pack(fill="x", pady=(0, 8))
        res = tk.Label(b, text="", font=fnt(9, "bold"), fg=T.MUTED, bg=T.CARD, justify="left", anchor="w", wraplength=430)
        bt = tk.Frame(b, bg=T.CARD)
        bt.pack(fill="x")
        PillButton(bt, "Charger le défi", lambda dd=d: self.charger_defi(dd), icon="model", kind="tonal",
                   tip="Charge le modèle de départ de ce défi dans l'écran Modèle.").pack(side="left")
        PillButton(bt, "Tester ma solution", lambda dd=d: self.tester_defi(dd), icon="flask",
                   tip="Lance 8 simulations avec les paramètres courants et attribue des étoiles.").pack(side="left", padx=8)
        res.pack(fill="x", pady=(8, 0))
        self.defi_ui[d["id"]] = dict(stars=etoiles, res=res)
        self.dessiner_etoiles(d["id"])

    def dessiner_etoiles(self, did):
        c = self.defi_ui[did]["stars"]
        c.delete("all")
        n = self.etoiles.get(did, 0)
        for i in range(3):
            draw_icon(c, "star", 14 + i * 28, 13, 22, "#F59E0B" if i < n else "#D5DBE6")

    def charger_defi(self, d):
        self.defi_actif = d["id"]
        base = dict(d["base"])
        self.p = params_preset(base)
        self.p["horizon"], self.p["warmup"] = 3000.0, 200.0
        self.sync_widgets()
        self.maj_modele()
        self.lbl_defi.config(text=f"Défi en cours : {d['titre']} — {d['objectif']}")
        self.lbl_defi.pack(fill="x", pady=(8, 0), before=self.chips_preset)
        self.afficher("modele")
        self.toast(f"Défi chargé : {d['titre']}")

    def tester_defi(self, d):
        if self.occupe:
            self.toast("Un calcul est déjà en cours")
            return
        ui = self.defi_ui[d["id"]]
        if self.defi_actif != d["id"]:
            ui["res"].config(text="Chargez d'abord ce défi, puis ajustez le modèle.", fg=T.RED)
            return
        ok, txt = d["contrainte"](self.p)
        if not ok:
            ui["res"].config(text=f"Contrainte non respectée : {txt}", fg=T.RED)
            return
        params = dict(self.p, horizon=3000.0, warmup=200.0)
        ui["res"].config(text="Simulation en cours…", fg=T.MUTED)

        def fin(etat):
            val = agreger(etat["ms"])[d["metrique"]][0]
            etoiles = sum(1 for s in d["niveaux"] if val <= s)
            self.etoiles[d["id"]] = max(self.etoiles.get(d["id"], 0), etoiles)
            pct = d["metrique"] in ("p_block", "p_renege")
            vtxt = f"{val * 100:.2f} %" if pct else f"{val:.3f}"
            msg = {0: "Pas encore… ", 1: "Bien ! ", 2: "Très bien ! ", 3: "Parfait ! "}[etoiles]
            ui["res"].config(text=f"{msg}Résultat : {vtxt} → {'★' * etoiles}{'☆' * (3 - etoiles)}",
                             fg=T.PRIMARY_D if etoiles else T.RED)
            self.dessiner_etoiles(d["id"])
            self.maj_etoiles()
            if etoiles == 3:
                self.toast(f"Défi « {d['titre']} » réussi à la perfection !")
        self.batch(params, 8, lambda a, b: None, fin)

    # ------------------------------------------------------------------
    #  Aide
    # ------------------------------------------------------------------
    def construire_aide(self):
        p = self.pages["aide"] = tk.Frame(self.conteneur, bg=T.BG)
        haut = tk.Frame(p, bg=T.BG)
        haut.pack(fill="x", padx=14, pady=(14, 8))
        self.chips_aide = ChipGroup(haut, AIDE_TOPICS, "demarrage", self.ecrire_aide,
                                    {k: f"Afficher : {AIDE_TITRES[k]}" for k, _ in AIDE_TOPICS})
        self.chips_aide.pack(side="left", fill="x", expand=True)
        PillButton(haut, "À propos", self.a_propos, icon="info", kind="tonal", tip="Informations sur l'application"
                   ).pack(side="right", padx=(10, 0))
        c = Card(p, pad=18, radius=18, expand=True)
        c.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        self.aide_txt = tk.Text(c.body, wrap="word", state="disabled", bg=T.CARD, fg=T.TEXT, relief="flat", bd=0,
                                highlightthickness=0, padx=6, pady=4, font=fnt(10), cursor="arrow")
        sb = tk.Scrollbar(c.body, orient="vertical", command=self.aide_txt.yview, width=10, relief="flat",
                          troughcolor=T.CARD, bd=0)
        self.aide_txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.aide_txt.pack(side="left", fill="both", expand=True)
        t = self.aide_txt
        t.tag_configure("h1", font=fnt(17, "bold"), foreground=T.TEXT, spacing3=8)
        t.tag_configure("h2", font=fnt(11, "bold"), foreground=T.PRIMARY, spacing1=14, spacing3=3)
        t.tag_configure("p", font=fnt(10), foreground=T.TEXT, spacing3=5, lmargin1=2, lmargin2=2)
        t.tag_configure("li", font=fnt(10), foreground=T.TEXT, spacing3=3, lmargin1=10, lmargin2=26)
        t.tag_configure("code", font=(MONO, 9), foreground="#0F172A", background="#F1F5F9", lmargin1=10, lmargin2=10,
                        spacing1=4, spacing3=6)
        self.ecrire_aide("demarrage")

    def ecrire_aide(self, topic):
        self.chips_aide.set(topic)
        t = self.aide_txt
        t.configure(state="normal")
        t.delete("1.0", "end")
        for tag, texte in contenu_aide(topic):
            t.insert("end", texte + "\n", tag)
        t.configure(state="disabled")
        t.yview_moveto(0)

    def ouvrir_aide(self, topic="demarrage"):
        self.afficher("aide")
        self.ecrire_aide(topic)

    def menu_aide(self):
        m = tk.Menu(self.root, tearoff=0, font=fnt(10), bg=T.CARD, fg=T.TEXT, bd=0, relief="flat",
                    activebackground=T.PRIMARY_L, activeforeground=T.PRIMARY_D)
        for cle, _ in AIDE_TOPICS:
            m.add_command(label="   " + AIDE_TITRES[cle] + "   ", command=lambda k=cle: self.ouvrir_aide(k),
                          accelerator="F1" if cle == "demarrage" else "")
        m.add_separator()
        m.add_command(label="   À propos   ", command=self.a_propos)
        x = self.btn_more.winfo_rootx()
        y = self.btn_more.winfo_rooty() + self.btn_more.winfo_height()
        try:
            m.tk_popup(x, y)
        finally:
            m.grab_release()

    def a_propos(self):
        d = tk.Toplevel(self.root)
        d.title("À propos")
        d.configure(bg=T.CARD)
        d.resizable(False, False)
        d.transient(self.root)
        logo = tk.Canvas(d, width=72, height=72, bg=T.CARD, highlightthickness=0)
        logo.pack(pady=(24, 8), padx=48)
        draw_logo(logo, 72)
        tk.Label(d, text="Files d'attente", font=fnt(16, "bold"), fg=T.TEXT, bg=T.CARD).pack()
        tk.Label(d, text="Laboratoire de simulation", font=fnt(10), fg=T.MUTED, bg=T.CARD).pack()
        tk.Label(d, text="Simulation à événements discrets · théorie des files\n"
                         "M/M/1, M/G/1/K, G/G/c, Erlang-B… et leurs variantes\nTkinter, bibliothèque standard",
                 font=fnt(9), fg=T.MUTED, bg=T.CARD, justify="center").pack(pady=(10, 14), padx=20)
        PillButton(d, "Fermer", d.destroy, kind="filled", tip="Fermer cette fenêtre").pack(pady=(0, 22))
        d.update_idletasks()
        x = self.root.winfo_rootx() + (self.root.winfo_width() - d.winfo_width()) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - d.winfo_height()) // 3
        d.geometry(f"+{x}+{y}")
        d.grab_set()

    # ------------------------------------------------------------------
    #  Toast et raccourcis
    # ------------------------------------------------------------------
    def toast(self, texte, ms=2800):
        self.toast_lbl.config(text=texte)
        self.toast_lbl.place(relx=.5, rely=1.0, y=-86, anchor="s")
        self.toast_lbl.lift()
        if self._toast_job:
            self.root.after_cancel(self._toast_job)
        self._toast_job = self.root.after(ms, self.toast_lbl.place_forget)

    def lier_raccourcis(self):
        r = self.root

        def sur_sim(fn):
            return lambda e: fn() if self.page in ("sim", "suivi") else None
        r.bind("<space>", sur_sim(self.fab_action))
        r.bind("<Key-s>", sur_sim(self.pas_sim))
        r.bind("<Right>", sur_sim(self.pas_sim))
        r.bind("<Key-f>", sur_sim(self.fin_rapide))
        r.bind("<Key-r>", sur_sim(self.nouvelle_sim_btn))
        r.bind("<Escape>", lambda e: self.nouvelle_sim_btn() if self.page in ("sim", "suivi") else None)
        r.bind("<Control-e>", lambda e: self.exporter_csv())
        r.bind("<Control-s>", lambda e: self.enregistrer_journal())
        r.bind("<F1>", lambda e: self.ouvrir_aide("demarrage"))
        for i, (cle, *_) in enumerate(self.PAGES, 1):
            r.bind(f"<Key-{i}>", lambda e, k=cle: self.afficher(k))
        for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            r.bind_all(ev, self.molette, add="+")


def main():
    racine = tk.Tk()
    init_theme(racine)
    racine.geometry("1220x860")
    racine.minsize(1080, 800)
    App(racine)
    racine.mainloop()


if __name__ == "__main__":
    main()
