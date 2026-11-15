"""Study tools that need no target column: explore, compare two columns, find groups.
Built on minilearn's helpers and numpy only."""
import numpy as np

import minilearn as ml


def _num(rows, i):
    return [float(r[i]) if ml.is_number(r[i]) and not ml.is_missing(r[i]) else np.nan for r in rows]


def overview(header, rows, info):
    cols, nums = [], {}
    for i, c in enumerate(info):
        if c["type"] == "numeric":
            raw = _num(rows, i)
            v = np.array(raw)
            v = v[~np.isnan(v)]
            nums[c["name"]] = raw
            cols.append(dict(name=c["name"], kind="numeric", mean=float(v.mean()), median=float(np.median(v)),
                             std=float(v.std()), min=float(v.min()), max=float(v.max()),
                             hist=np.histogram(v, bins=8)[0].tolist()))
        elif c["type"] == "text":
            u, n = np.unique([r[i] for r in rows if not ml.is_missing(r[i])], return_counts=True)
            top = np.argsort(-n)[:5]
            cols.append(dict(name=c["name"], kind="text", top=[[str(u[j]), int(n[j])] for j in top]))
    pairs, names = [], list(nums)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            x, y = np.array(nums[names[a]]), np.array(nums[names[b]])
            m = ~np.isnan(x) & ~np.isnan(y)
            if m.sum() > 2 and x[m].std() > 0 and y[m].std() > 0:
                pairs.append([names[a], names[b], float(np.corrcoef(x[m], y[m])[0, 1])])
    pairs.sort(key=lambda p: -abs(p[2]))
    return dict(columns=cols, pairs=pairs[:6])


def compare(header, rows, info, a, b):
    ia, ib = header.index(a), header.index(b)
    ta, tb = info[ia]["type"], info[ib]["type"]
    if ia == ib:
        raise ValueError("Pick two different columns to compare.")
    if ta == "numeric" and tb == "numeric":
        x, y = np.array(_num(rows, ia)), np.array(_num(rows, ib))
        m = ~np.isnan(x) & ~np.isnan(y)
        if m.sum() < 3 or x[m].std() == 0 or y[m].std() == 0:
            raise ValueError("These columns do not vary enough to compare.")
        r = float(np.corrcoef(x[m], y[m])[0, 1])
        slope = float(np.polyfit(x[m], y[m], 1)[0])
        return dict(kind="numeric_numeric", a=a, b=b, n=int(m.sum()), r=r, slope=slope)
    if {ta, tb} == {"numeric", "text"}:
        ni, ci = (ia, ib) if ta == "numeric" else (ib, ia)
        groups = {}
        for r, v in zip(rows, _num(rows, ni)):
            if not np.isnan(v) and not ml.is_missing(r[ci]):
                groups.setdefault(r[ci], []).append(v)
        allv = np.array([v for g in groups.values() for v in g])
        top = sorted(groups.items(), key=lambda kv: -len(kv[1]))[:8]
        ss_total = float(((allv - allv.mean()) ** 2).sum())
        ss_between = float(sum(len(g) * (np.mean(g) - allv.mean()) ** 2 for g in groups.values()))
        return dict(kind="numeric_text", number=header[ni], group=header[ci],
                    eta2=ss_between / ss_total if ss_total else 0.0,
                    rows=[[k, len(g), float(np.mean(g)), float(np.std(g)), float(np.median(g))] for k, g in top])
    if ta == "text" and tb == "text":
        rl = [k for k, _ in sorted(_count(rows, ia).items(), key=lambda kv: -kv[1])[:6]]
        cl = [k for k, _ in sorted(_count(rows, ib).items(), key=lambda kv: -kv[1])[:6]]
        grid = [[sum(1 for r in rows if r[ia] == p and r[ib] == q) for q in cl] for p in rl]
        return dict(kind="text_text", a=a, b=b, rows=rl, cols=cl, grid=grid)
    raise ValueError("One of these columns is empty.")


def _count(rows, i):
    out = {}
    for r in rows:
        if not ml.is_missing(r[i]):
            out[r[i]] = out.get(r[i], 0) + 1
    return out


def groups(header, rows, info, k):
    idx = []
    for i, c in enumerate(info):
        if c["type"] == "numeric":
            present = [v for v in (r[i] for r in rows) if not ml.is_missing(v)]
            if present and not ml.looks_like_id(present):
                idx.append(i)
    if len(idx) < 2:
        raise ValueError("Finding groups needs at least two numeric columns.")
    X = np.array([_num(rows, i) for i in idx]).T
    X = X[~np.isnan(X).any(axis=1)]
    if len(X) < k * 5:
        raise ValueError("Not enough complete rows for that many groups. Try fewer groups.")
    sd = X.std(axis=0)
    Z = (X - X.mean(axis=0)) / np.where(sd == 0, 1, sd)
    C = Z[np.random.default_rng(0).choice(len(Z), k, replace=False)]
    for _ in range(50):
        lab = ((Z[:, None] - C[None]) ** 2).sum(axis=2).argmin(axis=1)
        C = np.array([Z[lab == j].mean(axis=0) if (lab == j).any() else C[j] for j in range(k)])
    return dict(features=[header[i] for i in idx], rows_used=int(len(X)),
                overall=X.mean(axis=0).tolist(),
                clusters=[dict(size=int((lab == j).sum()), means=X[lab == j].mean(axis=0).tolist())
                          for j in range(k) if (lab == j).any()])