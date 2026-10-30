# Celestial Dome Octas

An interactive 3D trainer for estimating **cloud amount in octas** (eighths
of the celestial dome). It is a static site, so it runs in any modern browser
and is hosted on GitHub Pages without a build step.

## What it does

- **Octas tab:** shows the dome with 0–8 octas of cloud as a slice (an
  orange cut from horizon to zenith), a vert slice (the orange on its side:
  180° slices from the north horizon to the south, sliding up from one
  horizon, over the zenith, to the other), a round patch, a horizon ring, clumps or random patches. The *Deform*
  controls change the outline while the covered area stays exactly fixed.
- **Outside / Inside views:** orbit the translucent dome from outside, or
  stand at the observer's position and look around.
- **All-sky map:** a flat map of the whole dome. In the equal-area
  projection, area on the map is proportional to octas.
- **Cloud layer tab:** one flat cloud layer in the low, mid or high étage,
  either cumuliform (separate elements) or stratiform (a sheet with gaps).
  Coverage is picked in octas as seen on the dome; the clouds are sized to
  give it. Each étage tints its clouds. The scene is to scale and follows
  the Earth's curve: an observer with eyes 1.7 m up, a dome whose radius is
  the distance to their horizon (4.65 km), the ground inside it, and the
  layer drawn out to the observer's line of sight. The GPU draws the
  layer's projection on the dome directly, pixel by pixel, from the same
  random layout used for the octa numbers.
- **Quiz tab:** random skies to estimate, scored using the octa rounding rules.
- **Notes tab:** the geometry, explained for a beginner observer in Canada.

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
- `js/layer.js`: cloud-layer ray caster
- `js/main.js`: 3D scene, all-sky map and UI

## Caveat

The rules in the Notes tab summarise common WMO / North American practice.
For anything that matters, MANOBS (ECCC) and your instructor are the
authority.
