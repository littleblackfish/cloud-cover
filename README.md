# Celestial Dome Octas

An interactive 3D trainer for estimating **cloud amount in octas** (eighths
of the celestial dome). It is a static site, so it runs in any modern browser
and is hosted on GitHub Pages without a build step.

## What it does

- **Celestial dome tab:** the dome with 0–8 octas of cloud as a slice (an
  orange lying on its side, sliced between opposite horizon points) or a
  patch. Azimuth sets the horizon the cloud starts from and elevation
  (0–180°) how far it has climbed, over the zenith and down the far side.
  Plain geometry only; the covered area stays exactly fixed.
- **Outside / Inside views:** orbit the translucent dome from outside, or
  stand at the observer's position and look around.
- **All-sky map:** a flat map of the whole dome. In the equal-area
  projection, area on the map is proportional to octas.
- **Cloud layer tab:** one flat round patch on a flat cloud layer in the
  low, mid or high étage, sized to cover exactly the chosen octas of the
  dome. Azimuth and elevation (shared with the Celestial dome tab) place it.
  The patch can be drawn as one of the cloud types reported in Canadian
  METARs: CI, CC, CS, AC, AS, NS, SC, ST, SF and CF so far (ACC, CU, TCU and
  CB need height and come next). Each type's pattern lies on the layer, sized
  so its elements look the right size from the ground (CC grains under 1°,
  AC 1–5°, SC over 5°), so it crowds together toward the horizon. Picking a
  type moves the layer to its usual base, and the base slider covers the
  type's range; the étage follows from the base. Bases, base ranges and
  descriptions follow MANOBS (8th edition) table 7-1; the readout gives the
  amount term (FEW, SCT, BKN, OVC) from table 7-2.
  The scene is to scale and follows the Earth's curve: an observer with eyes
  1.7 m up, a dome whose radius is the distance to their horizon (4.65 km),
  the ground inside it, and the layer drawn out to the observer's line of
  sight. Each type has its own grey; a plain patch takes its étage's.
- **Quiz tab:** random plain slices and patches to estimate from inside the dome, scored using
  the octa rounding rules; the outside view and sky map open after answering.

## The key geometry

The fraction of the dome below elevation *e* is `sin e`, so:

| Octas of sky below | Elevation |
|---|---|
| 1 | 7.2° |
| 2 | 14.5° |
| 3 | 22.0° |
| 4 | **30.0°** |
| 5 | 38.7° |
| 6 | 48.6° |
| 7 | 61.0° |

Half the sky lies below 30°. A 45° orange slice from horizon to zenith is
always exactly one octa.

## Run locally

ES modules need a web server (opening the file directly won't work):

```sh
python3 -m http.server 8000
# then open http://localhost:8000
```

## Host on GitHub Pages

1. Merge into the default branch (`main`).
2. On GitHub: **Settings → Pages → Build and deployment → Source: Deploy from
   a branch**, branch `main`, folder `/ (root)`.
3. The site appears at `https://<user>.github.io/cloud-cover/`.

No internet access is needed at runtime: three.js is vendored in `vendor/`
(MIT licence, see `vendor/three-LICENSE`).

## Files

- `js/sky.js`: sky grid with solid-angle weights, shape fields, the
  equal-area threshold solver and octa rounding
- `js/layer.js`: cloud-layer field (shape + texture on the layer)
- `js/main.js`: 3D scene, all-sky map and UI

## Caveat

The tool shows the geometry of cloud amount. For observing and reporting
rules, MANOBS (ECCC) and your instructor are the authority.
