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
- **All-sky map and report:** a flat map of the whole dome (equal-area, so
  area on the map is proportional to octas), with the reported octas and the
  sky condition (e.g. `SCT044`) under it; hover for the exact amount.
- **Cloud layer tab:** one flat round patch on a flat cloud layer in the
  low, mid or high étage, sized to cover exactly the chosen octas of the
  dome. Azimuth and elevation (shared with the Celestial dome tab) place it.
  The patch can be drawn as one of the cloud types reported in Canadian
  METARs: CI, CC, CS, AC, ACC, AS, NS, SC, ST, SF, CU, CF, TCU and CB. Each
  type's pattern lies on the layer, sized
  so its elements look the right size from the ground (CC grains under 1°,
  AC 1–5°, SC over 5°), so it crowds together toward the horizon. CU, TCU and
  CB are solid 3D clouds (clusters of lumpy spheres with flat bases, lit by
  the sun; see js/cumulus.js): coverage is set by growing every cloud from
  its base, solved exactly per sky direction, so toward the horizon their
  sides fill the gaps. Near the observer (out to about 3° above the horizon), SC, AC,
  ACC, CF and SF are drawn as solid elements lined up with their pattern
  (ACC with turrets; the fractus types as torn clusters of small blobs),
  which takes over beyond. Picking a
  type moves the layer to its usual base (the median reported in Canadian
  METARs, 2023–24; its description gives the reported range and amounts), and the base slider covers the
  type's range; the étage follows from the base. Bases, base ranges and
  descriptions follow MANOBS (8th edition) table 7-1; the readout gives the
  amount term (FEW, SCT, BKN, OVC) from table 7-2.
  The scene is to scale and follows the Earth's curve: an observer with eyes
  1.7 m up, a dome whose radius is the distance to their horizon (4.65 km),
  the ground inside it, and the layer drawn out to the observer's line of
  sight. Each type has its own grey; a plain patch takes its étage's.
- **Sun:** set by season (equinox or solstice at 50° N) or by your location
  and any date, with a slider from sunrise to sunset (times from NOAA's solar
  formulas; local solar time without a location). It shows in the sky,
  behind the clouds, and lights the solid clouds, warmer and dimmer when low.
- **Quiz tab:** a random sky of one cloud type (any of the drawable types)
  at a base around its usual one, covering a random, usually fractional,
  amount. You start inside, looking straight up with the whole sky in view
  and no guides. A hint button gives the cloud base, as a ceilometer would.
  Answer the amount (and optionally the type); the answer is
  scored with the octa rounding rules and shown with its amount term, base
  and sky condition group (e.g. `BKN035`). The guides, outside view and sky
  map open after answering.

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
- `js/cumulus.js`: solid cumuliform clouds (layout, shapes, exact coverage)
- `js/sun.js`: the sun's position (declination, equation of time, sunrise and sunset)
- `js/climate.js`: how Canadian observers report each type (from METARs; see `docs/metar-stats.md` and `tools/`)
- `js/main.js`: 3D scene, all-sky map and UI

## Caveat

The tool shows the geometry of cloud amount. For observing and reporting
rules, MANOBS (ECCC) and your instructor are the authority.
