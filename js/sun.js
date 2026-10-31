// Where the sun is: NOAA's approximations for the sun's declination and the
// equation of time (good to a fraction of a degree and about a minute),
// from latitude, longitude, date and the hour angle.

const DEG = Math.PI / 180;

export function dayOfYear(date) {
  const start = Date.UTC(date.getUTCFullYear(), 0, 0);
  return Math.floor((date.getTime() - start) / 86400000);
}

// Declination (rad) and equation of time (minutes) on a day of the year.
export function solarDay(doy) {
  const g = (2 * Math.PI / 365) * (doy - 1);
  const eot = 229.18 * (0.000075 + 0.001868 * Math.cos(g) - 0.032077 * Math.sin(g)
    - 0.014615 * Math.cos(2 * g) - 0.040849 * Math.sin(2 * g));
  const decl = 0.006918 - 0.399912 * Math.cos(g) + 0.070257 * Math.sin(g) - 0.006758 * Math.cos(2 * g)
    + 0.000907 * Math.sin(2 * g) - 0.002697 * Math.cos(3 * g) + 0.00148 * Math.sin(3 * g);
  return { decl, eot };
}

// Hour angle (rad) of sunrise and sunset: the sun's upper edge on the
// horizon, with refraction (90.833 degrees from the zenith). Null when the
// sun does not set (or does not rise) that day.
export function halfDay(latDeg, decl) {
  const lat = latDeg * DEG;
  const c = Math.cos(90.833 * DEG) / (Math.cos(lat) * Math.cos(decl)) - Math.tan(lat) * Math.tan(decl);
  if (c <= -1) return Math.PI; // midnight sun
  if (c >= 1) return 0; // polar night
  return Math.acos(c);
}

// Elevation and azimuth (rad; azimuth from north, clockwise) at hour angle h
// (rad; negative in the morning).
export function sunPosition(latDeg, decl, h) {
  const lat = latDeg * DEG;
  const el = Math.asin(Math.sin(lat) * Math.sin(decl) + Math.cos(lat) * Math.cos(decl) * Math.cos(h));
  const az = Math.atan2(Math.sin(h), Math.cos(h) * Math.sin(lat) - Math.tan(decl) * Math.cos(lat)) + Math.PI;
  return { el, az: (az + 2 * Math.PI) % (2 * Math.PI) };
}

// Clock time (minutes after midnight UTC) of hour angle h at longitude lon
// (deg, east positive) on a day with equation of time eot.
export const utcMinutes = (h, lonDeg, eot) => 720 - 4 * lonDeg - eot + (h / DEG) * 4;
