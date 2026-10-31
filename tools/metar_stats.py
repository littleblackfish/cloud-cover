# Statistics of how Canadian observers report each cloud type, from METARs.
#
# 1. Download routine METARs (2023-2024) of staffed stations from the Iowa
#    Environmental Mesonet into /tmp/metar/<STATION>.csv, e.g.
#    https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=CYVR&data=metar&year1=2023&month1=1&day1=1&year2=2025&month2=1&day2=1&tz=Etc/UTC&format=onlycomma&latlon=no&missing=M&trace=T&direct=no&report_type=3
#    for CYVR CYYJ CYXS CYXY CYZF CYVQ CYFB CYYC CYEG CYMM CYXE CYQR CYWG CYQT
#    CYSB CYYZ CYXU CYOW CYUL CYQB CYQM CYHZ CYYT CYQX CYYQ.
# 2. python3 tools/metar_stats.py  (writes stats.json, prints tables)
# 3. python3 tools/metar_climate_js.py  (writes js/climate.js)
#
# Automated reports (AUTO) carry no cloud types and are skipped. Each sky
# group (FEW/SCT/BKN/OVC + base) is paired, in order, with the cloud-type
# remark (e.g. RMK SC4AC2) when their counts match and every remark code is
# a cloud type.
import csv, glob, re, json, collections, sys
TYPES = ['CI','CC','CS','ACC','AC','AS','NS','SC','ST','SF','TCU','CU','CF','CB']
SKY = re.compile(r'^(FEW|SCT|BKN|OVC)(\d{3})(CB|TCU)?$')
CODE = re.compile(r'(ACC|TCU|CI|CC|CS|AC|AS|NS|SC|ST|SF|CU|CF|CB|[A-Z]{2})(\d)')
single = collections.defaultdict(list)   # type -> [(base_ft, octas, station, month)]
paired = collections.defaultdict(list)
n_rep = n_auto = n_used = 0
for f in sorted(glob.glob('/tmp/metar/*.csv')):
    st = f.split('/')[-1][:4]
    for row in csv.DictReader(open(f)):
        m = row.get('metar') or ''
        if not m or m == 'M': continue
        n_rep += 1
        if ' AUTO ' in m or 'RMK' not in m: n_auto += 1; continue
        body, rmk = m.split(' RMK ', 1) if ' RMK ' in m else (m, '')
        groups = []
        for tok in body.split():
            g = SKY.match(tok)
            if g: groups.append((g.group(1), int(g.group(2)) * 100))
        if not groups: continue
        first = rmk.split()[0] if rmk.split() else ''
        codes = CODE.findall(first)
        if not codes or ''.join(a + b for a, b in codes) != first: continue
        if len(codes) != len(groups): continue
        if any(c not in TYPES for c, _ in codes): continue  # obscurations etc.
        n_used += 1
        month = int(row['valid'][5:7])
        for (term, base), (c, o) in zip(groups, codes):
            paired[c].append((base, int(o), st, month))
        if len(groups) == 1:
            c, o = codes[0]
            single[c].append((groups[0][1], int(o), st, month))
def pct(v, q):
    v = sorted(v); return v[min(len(v)-1, int(q/100*len(v)))] if v else None
out = {'reports': n_rep, 'auto_or_no_rmk': n_auto, 'used': n_used, 'types': {}}
for c in TYPES:
    for name, d in (('single', single), ('all', paired)):
        L = d[c]
        if not L: continue
        bases = [b for b, o, s, mo in L]; octs = [o for b, o, s, mo in L]
        hist = collections.Counter(octs)
        out['types'].setdefault(c, {})[name] = {
            'n': len(L),
            'base_pct': {q: pct(bases, q) for q in (1, 5, 25, 50, 75, 95, 99)},
            'octas': {k: round(100 * hist.get(k, 0) / len(L), 1) for k in range(1, 9)},
            'stations': len(set(s for b, o, s, mo in L)),
            'base_hist': collections.Counter((b // 500) * 500 for b in bases),
            'season': collections.Counter(mo for b, o, s, mo in L),
        }
json.dump(out, open('stats.json', 'w'), default=lambda x: dict(x))
print('reports', n_rep, 'auto/no rmk', n_auto, 'used', n_used)
print(f"{'type':4} {'n1':>6} {'nAll':>7}  base single: p5  p25  p50  p75  p95 | octas single 1..8 %")
for c in TYPES:
    t = out['types'].get(c)
    if not t: continue
    s = t.get('single'); a = t['all']
    if s:
        b = s['base_pct']; o = s['octas']
        print(f"{c:4} {s['n']:6} {a['n']:7}  {b[5]:6} {b[25]:6} {b[50]:6} {b[75]:6} {b[95]:6} | " + ' '.join(f"{o[k]:5.1f}" for k in range(1, 9)))
    else:
        print(f"{c:4} {0:6} {a['n']:7}")
print()
print('ALL layers (multi-layer included): base p5 p50 p95 | octas 1..8 %')
for c in TYPES:
    a = out['types'].get(c, {}).get('all')
    if not a: continue
    b = a['base_pct']; o = a['octas']
    print(f"{c:4} {a['n']:7} {b[5]:6} {b[50]:6} {b[95]:6} | " + ' '.join(f"{o[k]:5.1f}" for k in range(1, 9)))
