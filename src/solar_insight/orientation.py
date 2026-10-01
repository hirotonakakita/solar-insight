"""Solar axes for an image whose top points toward local zenith."""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import math
import cv2


def parse_observed_at(value, timezone_name="Asia/Tokyo"):
    """Naive ISO timestamps use the explicit timezone, never the PC timezone."""
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo(timezone_name))
        return dt.astimezone(timezone.utc)
    except (ValueError, KeyError) as exc:
        raise ValueError("Use an ISO observation timestamp and a valid timezone") from exc


def solar_orientation(observed_at, latitude, longitude, elevation_m=0,
                      timezone_name="Asia/Tokyo", mirror_x=False):
    if not all(math.isfinite(x) for x in (latitude, longitude, elevation_m)):
        raise ValueError("Observer coordinates must be finite")
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("Latitude/longitude out of range (degrees, east positive)")
    dt = parse_observed_at(observed_at, timezone_name)
    try:
        import astropy.units as u
        from astropy.coordinates import AltAz, EarthLocation, get_sun
        from astropy.time import Time
        from astropy.utils import iers
        from sunpy.coordinates import sun
    except ImportError as exc:
        raise ImportError("Solar axes require: python -m pip install -e '.[solar]'") from exc
    location = EarthLocation.from_geodetic(longitude*u.deg, latitude*u.deg,
                                           elevation_m*u.m)
    t = Time(dt)
    # Bundled IERS tables: no unexpected network download during analysis.
    with iers.conf.set_temp("auto_download", False):
        altitude = get_sun(t).transform_to(AltAz(obstime=t, location=location,
                                                pressure=0*u.hPa)).alt.to_value(u.deg)
        if altitude <= 0:
            raise ValueError("Sun is below the geometric horizon at the specified time/site")
        if altitude > 89.9:
            raise ValueError("Zenith-up orientation is ill-defined near the zenith")
        angle = float(sun.orientation(location, t).to_value(u.deg))
        p = float(sun.P(t).to_value(u.deg))
        b0 = float(sun.B0(t).to_value(u.deg))
    return {"observed_at_utc": dt.isoformat(), "input_timezone": timezone_name,
            "latitude_deg": latitude, "longitude_deg": longitude,
            "elevation_m": elevation_m, "image_top": "local_zenith",
            "mirror_x": mirror_x, "solar_north_from_zenith_deg": angle,
            "p_angle_deg": p, "b0_angle_deg": b0,
            "sun_altitude_deg": float(altitude),
            "angle_convention": "eastward from zenith; includes P angle"}


def axis_vectors(angle_deg, mirror_x=False):
    """Pixel vectors (x right, y down), unmirrored sky east is left."""
    a = math.radians(angle_deg)
    north = (-math.sin(a), -math.cos(a))
    west = (math.cos(a), -math.sin(a))
    if mirror_x:
        north = (-north[0], north[1])
        west = (-west[0], west[1])
    return {"N": north, "S": (-north[0], -north[1]),
            "W": west, "E": (-west[0], -west[1])}


def draw_disk_guides(image, center, radius, orientation=None):
    cv2.circle(image, center, radius, (0, 220, 255), 1, cv2.LINE_AA)
    if orientation is None:
        return
    vectors = axis_vectors(orientation["solar_north_from_zenith_deg"],
                           orientation["mirror_x"])
    def point(vector, distance):
        return tuple(int(round(c + distance*v)) for c, v in zip(center, vector))
    for first, second, color in (("N", "S", (255, 220, 0)),
                                  ("E", "W", (0, 180, 255))):
        cv2.line(image, point(vectors[first], radius), point(vectors[second], radius),
                 color, 1, cv2.LINE_AA)
        for label in (first, second):
            x, y = point(vectors[label], max(0, radius-24))
            cv2.putText(image, label, (max(0, min(image.shape[1]-18, x-7)),
                                      max(14, min(image.shape[0]-3, y+5))),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2, cv2.LINE_AA)
