import json
s = json.load(open('stats.json'))
out = {}
for c, t in s['types'].items():
    src = t.get('single') or t['all']
    hist = sorted((int(k), v) for k, v in src['base_hist'].items())
    out[c] = {
        'n': src['n'], 'nAll': t['all']['n'],
        'base': [src['base_pct'][str(q)] for q in (5, 25, 50, 75, 95)],
        'baseHist': hist,
        'octas': [src['octas'][str(k)] for k in range(1, 9)],
    }
js = """// How Canadian observers report each cloud type, from routine METARs of 25
// staffed stations, 2023-2024 (348,272 reports with cloud-type remarks; see
// tools/metar_stats.py). For each type, from reports with that type as the
// only layer: n reports; base percentiles 5, 25, 50, 75, 95 (ft); baseHist,
// counts per 500 ft of base; octas, the percent of reports with 1..8 octas.
// nAll counts the type in any layer. Generated: do not edit by hand.
export const CLIMATE = @@;
""".replace("@@", json.dumps(out, separators=(",", ":")))
open('js/climate.js', 'w').write(js)
print(len(js))
