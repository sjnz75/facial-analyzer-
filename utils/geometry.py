import math
import numpy as np

def line_angle_deg(p1, p2, horizontal=False):
    """
    Calcola l'angolo tra la linea p1-p2 e l'orizzontale (default) o la verticale.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    rad = math.atan2(dy, dx)
    deg = math.degrees(rad)
    if horizontal:
        return abs(deg)
    # per verticale, sottraggo 90°
    return abs(abs(deg) - 90)


def line_deviation_mm(gl, sn, pg, pixels_per_mm=3.78):
    """
    Distanza del punto sn dalla retta gl-pg (in mm).
    """
    # coeff angolari in pixel
    x0, y0 = sn
    x1, y1 = gl
    x2, y2 = pg
    num = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
    den = math.hypot(y2-y1, x2-x1)
    dist_px = num/den
    return dist_px / pixels_per_mm


def segment_length(p1, p2, pixels_per_mm=3.78):
    dist_px = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
    return dist_px / pixels_per_mm


def compute_vertical_thirds(tri, gl, sn, me, pixels_per_mm=3.78):
    """
    Restituisce percentuali dei tre terzi sul totale trichion->menton.
    """
    L_total = segment_length(tri, me, pixels_per_mm)
    L1 = segment_length(tri, gl, pixels_per_mm)
    L2 = segment_length(gl, sn, pixels_per_mm)
    L3 = segment_length(sn, me, pixels_per_mm)
    return [L1/L_total*100, L2/L_total*100, L3/L_total*100]
