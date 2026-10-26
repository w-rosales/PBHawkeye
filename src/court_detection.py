"""Local court registration using painted-line evidence and a pickleball template.

The detector targets a full court filmed from a stationary camera behind a baseline.
Color provides an optional search region; the final fit must explain the painted
baselines, sidelines, kitchen lines and service centerlines in perspective.
"""

from itertools import combinations
from typing import List, Tuple

import cv2
import numpy as np


# Coordinates run from the near-left corner (0, 0) to the far-right (20, 44).
COURT_LINES = np.float32([
    [[0, 44], [20, 44]], [[20, 44], [20, 0]],
    [[20, 0], [0, 0]], [[0, 0], [0, 44]],
    [[0, 15], [20, 15]], [[0, 29], [20, 29]],
    [[10, 0], [10, 15]], [[10, 29], [10, 44]],
])
COURT_CORNERS = np.float32([[0, 44], [20, 44], [20, 0], [0, 0]])


def _surface_region(frame):
    """Find the connected, saturated playing surface near the lower image center."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    patch = hsv[int(.35*h):int(.90*h), int(.22*w):int(.78*w)]
    colors = patch[..., 0][(patch[..., 1] > 45) & (patch[..., 2] > 40)]
    if colors.size < .01*w*h:
        return None
    histogram = np.bincount(colors, minlength=180).astype(float)
    histogram = np.convolve(np.r_[histogram[-4:], histogram, histogram[:4]], np.ones(9), 'valid')
    hue = int(np.argmax(histogram))
    delta = np.abs(hsv[..., 0].astype(float) - hue)
    delta = np.minimum(delta, 180-delta)
    mask = np.uint8((delta < 15) & (hsv[..., 1] > 35) & (hsv[..., 2] > 30))*255
    size = max(5, int(w/90) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((size, size), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    best, best_score = None, 0
    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < .035*w*h:
            continue
        anchor = np.count_nonzero(labels[int(.45*h):int(.90*h), int(.35*w):int(.65*w)] == i)
        score = anchor + area*.1
        if score > best_score:
            best, best_score = i, score
    if best is None or stats[best, cv2.CC_STAT_AREA] > .85*w*h:
        return None
    ax, ay, aw, ah = stats[best, cv2.CC_STAT_LEFT], stats[best, cv2.CC_STAT_TOP], stats[best, cv2.CC_STAT_WIDTH], stats[best, cv2.CC_STAT_HEIGHT]
    court_mask = np.uint8(labels == best) * 255
    for i in range(1, count):
        if i == best:
            continue
        area = stats[i, cv2.CC_STAT_AREA]
        if area < .008*w*h:
            continue
        cx, cy, cw, ch = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]

        is_far_court = (cy >= int(.18*h) and cy + ch <= ay + int(.08*h)
                        and ay - (cy + ch) < int(.20*h)
                        and cx >= ax - int(.06*w) and cx + cw <= ax + aw + int(.06*w))

        is_side_split = (abs(cy - ay) < int(.12*h) and abs((cy + ch) - (ay + ah)) < int(.12*h)
                         and (abs(cx - (ax + aw)) < int(.08*w) or abs(ax - (cx + cw)) < int(.08*w)))

        if is_far_court or is_side_split:
            court_mask[labels == i] = 255

    contours, _ = cv2.findContours(court_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    hull = cv2.convexHull(np.vstack(contours))
    region = np.zeros_like(mask)
    cv2.fillConvexPoly(region, hull, 255)
    return region


def court_line_evidence(frame):
    """Return a thin bright-line mask, optionally restricted to the court surface."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    size = max(11, int(w*.018) | 1)
    ridge = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    # Weathered paint inherits the blue surface hue in shadow. Requiring almost
    # zero saturation discards the near kitchen line in real outdoor footage.
    saturation = hsv[..., 1].astype(np.float32)
    local_saturation = cv2.GaussianBlur(saturation, (0, 0), max(2., w*.007))
    paint_color = (saturation < 155) | (saturation < local_saturation-18)
    mask = np.uint8((ridge > 11) & paint_color & (hsv[..., 2] > 65))*255
    region = _surface_region(frame)
    if region is not None:
        margin = max(5, int(w*.012))
        search = cv2.dilate(region, np.ones((margin*2+1, margin*2+1), np.uint8))
        mask = cv2.bitwise_and(mask, search)
    # A rear-court view has no playing surface in the very top of the image.
    mask[:int(.12*h)] = 0
    return mask, region


def _line_candidates(mask):
    h, w = mask.shape
    segments = cv2.HoughLinesP(mask, 1, np.pi/720, threshold=max(18, int(w*.024)),
                              minLineLength=max(20, int(w*.04)), maxLineGap=max(8, int(w*.018)))
    if segments is None:
        return [], []
    groups = [[], []]
    for x1, y1, x2, y2 in segments.reshape(-1, 4):
        p, q = np.float64([x1, y1]), np.float64([x2, y2])
        length = float(np.linalg.norm(q-p))
        line = np.cross(np.r_[p, 1.], np.r_[q, 1.])
        line /= np.linalg.norm(line[:2])
        if line[0] < 0:
            line = -line
        transverse = abs(y2-y1) < .45*abs(x2-x1)
        groups[0 if transverse else 1].append((line, length, p, q))
    result = []
    for candidates in groups:
        clusters = []
        for line, length, p, q in sorted(candidates, key=lambda item: -item[1]):
            for cluster in clusters:
                # Merge fragments of the same marking, preserving camera roll.
                if abs(np.dot(line[:2], cluster[0][:2])) > .998 and max(
                        abs(np.dot(cluster[0], np.r_[p, 1.])),
                        abs(np.dot(cluster[0], np.r_[q, 1.]))) < max(3, w*.004):
                    cluster[1] += length
                    cluster[2].extend([p, q])
                    break
            else:
                clusters.append([line, length, [p, q]])
        fitted = []
        for line, support, points in clusters:
            vx, vy, x, y = cv2.fitLine(np.float32(points), cv2.DIST_HUBER, 0, .01, .01).ravel()
            fitted.append((np.float64([vy, -vx, vx*y-vy*x]), support))
        result.append(sorted(fitted, key=lambda item: -item[1])[:16])
    return result[0], result[1]


def _intersection(a, b):
    p = np.cross(a, b)
    if abs(p[2]) < 1e-7:
        return None
    return p[:2]/p[2]


def _valid_quad(quad, width, height):
    if quad is None or not np.isfinite(quad).all():
        return False
    if np.any(quad < [-.04*width, -.04*height]) or np.any(quad > [1.04*width, 1.04*height]):
        return False
    polygon = np.float32(quad).reshape(-1, 1, 2)
    if not cv2.isContourConvex(polygon) or cv2.contourArea(polygon) < .045*width*height:
        return False
    far, near = np.linalg.norm(quad[1]-quad[0]), np.linalg.norm(quad[2]-quad[3])
    left, right = np.linalg.norm(quad[3]-quad[0]), np.linalg.norm(quad[2]-quad[1])
    near_y = (quad[2, 1] + quad[3, 1]) / 2
    near_cx = (quad[2, 0] + quad[3, 0]) / 2
    return (quad[0, 0] < quad[1, 0] and quad[3, 0] < quad[2, 0]
            and np.mean(quad[:2, 1]) + .15*height < np.mean(quad[2:, 1])
            and near_y > .55*height
            and .15*width < near_cx < .85*width
            and far > .10*width and near > .25*width and .12 < far/near <= 1.05
            and min(left, right) > .20*height and min(left, right)/max(left, right) > .35)


def _sample_template(quad, samples=60):
    matrix = cv2.getPerspectiveTransform(COURT_CORNERS, np.float32(quad))
    alpha = np.linspace(.025, .975, samples, dtype=np.float32)
    points = COURT_LINES[:, :1] + (COURT_LINES[:, 1:] - COURT_LINES[:, :1])*alpha[None, :, None]
    return cv2.perspectiveTransform(points.reshape(-1, 1, 2), matrix).reshape(8, samples, 2)


def _score_quad(quad, distance, region=None):
    h, w = distance.shape
    projected = _sample_template(quad)
    xy = np.rint(projected).astype(int)
    inside = (xy[..., 0] >= 0) & (xy[..., 0] < w) & (xy[..., 1] >= 0) & (xy[..., 1] < h)
    x, y = np.clip(xy[..., 0], 0, w-1), np.clip(xy[..., 1], 0, h-1)
    distances = distance[y, x]
    tolerance = max(2., w*.0025)
    support = (np.exp(-.5*np.minimum(distances/tolerance, 12.)**2)*inside).mean(axis=1)
    # Internal lines disambiguate a baseline from a net tape or kitchen line.
    # The far kitchen marking is often hidden behind the raised net. The near
    # kitchen and service centerline remain useful independent constraints.
    score = (.48*support[:4].mean() + .23*support[4]
             + .06*support[5] + .16*support[6] + .07*support[7])
    score *= .6 + .4*float(np.min(support[:4]))
    if region is not None:
        center = np.mean(quad, axis=0).astype(int)
        if region[np.clip(center[1], 0, h-1), np.clip(center[0], 0, w-1)] == 0:
            score *= .5
    return float(score), support


def court_alignment_score(frame, corners):
    """Measure support for an existing court, without searching for a new one."""
    height, width = frame.shape[:2]
    scale = min(1., 960./width)
    small = cv2.resize(frame, (round(width*scale), round(height*scale)), interpolation=cv2.INTER_AREA)
    mask, region = court_line_evidence(small)
    distance = cv2.distanceTransform(cv2.bitwise_not(mask), cv2.DIST_L2, 3)
    return _score_quad(np.float32(corners)*scale, distance, region)[0]


def _refine_quad(quad, distance):
    """Align a candidate against all visible paint with a robust pixel residual."""
    from scipy.optimize import least_squares

    h, w = distance.shape
    original = quad.ravel().copy()
    weights = np.sqrt([2., 2., 2., 2., 1.4, .25, 1.2, .5])[:, None]

    def residual(values):
        points = _sample_template(values.reshape(4, 2), samples=80)
        # Bilinear sampling keeps the objective smooth below one pixel.
        x = np.clip(points[..., 0], 0, w-1.001)
        y = np.clip(points[..., 1], 0, h-1.001)
        ix, iy = x.astype(int), y.astype(int)
        fx, fy = x-ix, y-iy
        d = ((1-fx)*(1-fy)*distance[iy, ix] + fx*(1-fy)*distance[iy, ix+1]
             + (1-fx)*fy*distance[iy+1, ix] + fx*fy*distance[iy+1, ix+1])
        # Occluded paint must not drag the court toward a player or the net.
        limits = np.array([max(12., w*.015)]*4 + [max(6., w*.009)]*4)[:, None]
        d = np.minimum(d, limits)
        return np.r_[(d*weights).ravel(), .08*(values-original)]

    radius = max(8., w*.022)
    fit = least_squares(residual, original, bounds=(original-radius, original+radius),
                        loss='soft_l1', f_scale=1.5, diff_step=.0005, max_nfev=45)
    return fit.x.reshape(4, 2)


def _snap_boundaries(quad, mask):
    """Fit stripe centers so thick paint/occlusions cannot move an outer corner."""
    h, w = mask.shape
    radius = max(6, int(w*.015))
    offsets = np.arange(-radius, radius+1)
    lines = []
    for start, end in zip(quad, np.roll(quad, -1, axis=0)):
        vector = end-start
        normal = np.array([-vector[1], vector[0]])/np.linalg.norm(vector)
        points = []
        for alpha in np.linspace(.04, .96, 100):
            center = start + alpha*vector
            profile = np.rint(center + offsets[:, None]*normal).astype(int)
            valid = ((profile[:, 0] >= 0) & (profile[:, 0] < w)
                     & (profile[:, 1] >= 0) & (profile[:, 1] < h))
            white = np.zeros(len(offsets), dtype=bool)
            white[valid] = mask[profile[valid, 1], profile[valid, 0]] > 0
            edges = np.diff(np.r_[False, white, False].astype(int))
            runs = [(a, b) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
                    if b-a <= max(8, w*.012)]
            if runs:
                a, b = min(runs, key=lambda run: abs((offsets[run[0]]+offsets[run[1]-1])/2))
                offset = (offsets[a]+offsets[b-1])/2
                points.append(center+normal*offset)
        if len(points) < 25:
            line = np.cross(np.r_[start, 1.], np.r_[end, 1.])
        else:
            vx, vy, x, y = cv2.fitLine(np.float32(points), cv2.DIST_HUBER, 0, .01, .01).ravel()
            line = np.array([vy, -vx, vx*y-vy*x])
        lines.append(line)
    points = [_intersection(lines[i-1], lines[i]) for i in range(4)]
    return np.array(points) if all(point is not None for point in points) else quad


def detect_court(frame: np.ndarray) -> Tuple[List[Tuple[float, float]], float]:
    """Return four proposed corners and a measured [0, 1] line-match score.

    A zero score means no supported court was found; the four returned preset
    points are only handles for manual correction, never a successful detection.
    """
    height, width = frame.shape[:2]
    prior = [(width*.25, height*.25), (width*.75, height*.25),
             (width*.90, height*.90), (width*.10, height*.90)]
    scale = min(1., 1280./width)
    small = cv2.resize(frame, (round(width*scale), round(height*scale))) if scale < 1 else frame
    h, w = small.shape[:2]
    mask, region = court_line_evidence(small)
    if np.count_nonzero(mask) < w*.4:
        return prior, 0.
    transverse, longitudinal = _line_candidates(mask)
    if len(transverse) < 2 or len(longitudinal) < 2:
        return prior, 0.
    distance = cv2.distanceTransform(cv2.bitwise_not(mask), cv2.DIST_L2, 3)
    best_quad, best_score = None, 0.
    proposals = []
    for (a, _), (b, _) in combinations(longitudinal, 2):
        for (c, _), (d, _) in combinations(transverse, 2):
            pts = [_intersection(a, c), _intersection(b, c), _intersection(b, d), _intersection(a, d)]
            if any(p is None for p in pts):
                continue
            quad = np.array(pts)
            # Sort the two end lines in image y, then their intersections in x.
            top, bottom = quad[:2], quad[2:]
            if np.mean(top[:, 1]) > np.mean(bottom[:, 1]):
                top, bottom = bottom, top
            quad = np.array([*top[np.argsort(top[:, 0])], *bottom[np.argsort(-bottom[:, 0])]])
            if not _valid_quad(quad, w, h):
                continue
            score, _ = _score_quad(quad, distance, region)
            proposals.append((score, quad))
            if score > best_score:
                best_quad, best_score = quad, score
    if best_quad is None:
        return prior, 0.
    # Refine a small set of distinct hypotheses, rather than tuning only whichever
    # Hough segment happened to be longest or landed on one edge of thick paint.
    refined_seeds = []
    for score, quad in sorted(proposals, key=lambda item: -item[0]):
        if any(np.linalg.norm(quad-seed, axis=1).max() < w*.008 for seed in refined_seeds):
            continue
        refined_seeds.append(quad)
        refined = _refine_quad(quad, distance)
        if _valid_quad(refined, w, h):
            refined_score, _ = _score_quad(refined, distance, region)
            if refined_score > best_score:
                best_quad, best_score = refined, refined_score
        if len(refined_seeds) >= 8:
            break
    snapped = _snap_boundaries(best_quad, mask)
    if _valid_quad(snapped, w, h):
        snapped_score, _ = _score_quad(snapped, distance, region)
        if snapped_score >= best_score-.06:
            best_quad, best_score = snapped, snapped_score
    return [(float(x/scale), float(y/scale)) for x, y in best_quad], best_score
