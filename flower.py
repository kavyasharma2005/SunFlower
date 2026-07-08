"""
flower.py
Holds the flower's growth state (stem/leaves/petals) and renders it as a
detailed sunflower: a two-layer ring of tapered, gradient-shaded petals
around a textured seed-spiral center, with a soft glow and shaded stem/leaves.
"""

import math
import numpy as np
import cv2

# BGR colors
PETAL_BASE = (10, 130, 235)     # deeper orange, near the center
PETAL_TIP = (25, 210, 255)      # bright golden yellow, at the tip
PETAL_BASE_INNER = (10, 105, 200)
PETAL_TIP_INNER = (20, 180, 250)

CENTER_OUTER = (25, 55, 85)      # dark brown ring
CENTER_INNER = (35, 75, 110)     # slightly lighter brown
SEED_COLOR = (15, 35, 60)        # near-black brown seed dots
CENTER_RIM = (10, 25, 45)

STEM_COLOR = (50, 145, 55)
STEM_HIGHLIGHT = (80, 190, 90)
LEAF_BASE = (35, 110, 45)
LEAF_TIP = (60, 165, 70)
LEAF_VEIN = (20, 80, 30)

GOLDEN_ANGLE = math.pi * (3 - math.sqrt(5))  # ~137.5 degrees, in radians


def _soft_glow(canvas, center, radius, color, alpha):
    """Blend a filled circle at partial opacity to fake a soft glow (cheap, no blur)."""
    if radius <= 0:
        return
    tmp = canvas.copy()
    cv2.circle(tmp, center, radius, color, -1, cv2.LINE_AA)
    cv2.addWeighted(tmp, alpha, canvas, 1 - alpha, 0, dst=canvas)


def _draw_petal(canvas, base, tip, max_width, color_base, color_tip):
    """Draw one tapered, gradient-shaded petal from base to tip."""
    bx, by = base
    tx, ty = tip
    dx, dy = tx - bx, ty - by
    length = math.hypot(dx, dy)
    if length < 1:
        return
    ux, uy = dx / length, dy / length
    px, py = -uy, ux  # perpendicular unit vector

    segments = 10
    left_pts, right_pts = [], []
    for i in range(segments + 1):
        t = i / segments
        # widest near the base, tapering smoothly to a point at the tip,
        # with a slight outward belly for a more natural petal silhouette
        w = max_width * (1 - t) * (1 + 0.3 * math.sin(math.pi * t))
        cx, cy = bx + dx * t, by + dy * t
        half_w = w / 2
        left_pts.append((int(cx + px * half_w), int(cy + py * half_w)))
        right_pts.append((int(cx - px * half_w), int(cy - py * half_w)))

    # base fill (deeper/orange tone)
    poly = np.array(left_pts + right_pts[::-1], dtype=np.int32)
    cv2.fillPoly(canvas, [poly], color_base, lineType=cv2.LINE_AA)

    # lighter highlight over the outer half toward the tip for a gradient look
    mid = segments // 2
    tip_poly = np.array(left_pts[mid:] + right_pts[mid:][::-1], dtype=np.int32)
    if len(tip_poly) >= 3:
        cv2.fillPoly(canvas, [tip_poly], color_tip, lineType=cv2.LINE_AA)

    # subtle center vein for definition
    shade = tuple(max(0, c - 35) for c in color_base)
    cv2.line(canvas, base, tip, shade, 1, cv2.LINE_AA)


def _draw_seed_center(canvas, center, radius):
    """Textured sunflower center using a Vogel (golden-angle) spiral of seed dots."""
    cx, cy = center
    cv2.circle(canvas, (cx, cy), radius, CENTER_OUTER, -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), int(radius * 0.88), CENTER_INNER, -1, cv2.LINE_AA)

    num_seeds = max(20, int(radius * 2.2))
    seed_r = max(1, int(radius * 0.055))
    for i in range(num_seeds):
        r = radius * 0.85 * math.sqrt(i / num_seeds)
        theta = i * GOLDEN_ANGLE
        sx = int(cx + r * math.cos(theta))
        sy = int(cy + r * math.sin(theta))
        cv2.circle(canvas, (sx, sy), seed_r, SEED_COLOR, -1, cv2.LINE_AA)

    cv2.circle(canvas, (cx, cy), radius, CENTER_RIM, 2, cv2.LINE_AA)


class Flower:
    def __init__(self, base_point, max_stem_height=260, max_petals=13, max_leaves=4):
        self.base_x, self.base_y = base_point
        self.max_stem_height = max_stem_height
        self.max_petals = max_petals
        self.max_leaves = max_leaves

        self.stem_height = 0.0
        self.petal_progress = 0.0  # 0..max_petals, fractional
        self.leaf_progress = 0.0   # 0..max_leaves, fractional

        self.stem_growth_rate = 2.2
        self.petal_growth_rate = 0.05
        self.leaf_growth_rate = 0.035

    # --- growth actions, called once per frame while a gesture is held ---
    def grow_stem(self):
        self.stem_height = min(self.max_stem_height, self.stem_height + self.stem_growth_rate)

    def grow_petals(self):
        if self.stem_height >= self.max_stem_height * 0.35:
            self.petal_progress = min(self.max_petals, self.petal_progress + self.petal_growth_rate)

    def grow_leaves(self):
        if self.stem_height >= self.max_stem_height * 0.2:
            self.leaf_progress = min(self.max_leaves, self.leaf_progress + self.leaf_growth_rate)

    def reset(self):
        self.stem_height = 0.0
        self.petal_progress = 0.0
        self.leaf_progress = 0.0

    def is_fully_bloomed(self):
        return (
            self.stem_height >= self.max_stem_height
            and self.petal_progress >= self.max_petals
            and self.leaf_progress >= self.max_leaves
        )

    def _tip_point(self):
        sway = 14 * math.sin(self.stem_height / self.max_stem_height * math.pi)
        return int(self.base_x + sway), int(self.base_y - self.stem_height)

    def draw(self, frame):
        overlay = frame.copy()

        # --- Stem, drawn as segments so it can gently sway, with a highlight stripe ---
        segments = 24
        prev = (self.base_x, self.base_y)
        for i in range(1, segments + 1):
            t = i / segments
            h = self.stem_height * t
            sway = 14 * math.sin(h / self.max_stem_height * math.pi) * t
            pt = (int(self.base_x + sway), int(self.base_y - h))
            cv2.line(overlay, prev, pt, STEM_COLOR, 9, cv2.LINE_AA)
            cv2.line(overlay, prev, pt, STEM_HIGHLIGHT, 2, cv2.LINE_AA)
            prev = pt

        # --- Leaves, two-tone with a center vein ---
        full_leaves = int(self.leaf_progress)
        partial = self.leaf_progress - full_leaves
        for i in range(self.max_leaves):
            growing = i < full_leaves or (i == full_leaves and partial > 0)
            if not growing:
                continue
            size_frac = 1.0 if i < full_leaves else max(0.15, partial)
            t = (i + 1) / (self.max_leaves + 1)
            h = self.stem_height * t
            sway = 14 * math.sin(h / self.max_stem_height * math.pi) * t
            attach_x = int(self.base_x + sway)
            attach_y = int(self.base_y - h)
            side = 1 if i % 2 == 0 else -1
            leaf_w = int(40 * size_frac)
            leaf_h = int(19 * size_frac)
            center = (attach_x + side * leaf_w, attach_y)
            angle = 30 * side
            cv2.ellipse(overlay, center, (leaf_w, leaf_h), angle, 0, 360, LEAF_BASE, -1, cv2.LINE_AA)
            cv2.ellipse(overlay, center, (int(leaf_w * 0.6), int(leaf_h * 0.6)), angle, 0, 360, LEAF_TIP, -1, cv2.LINE_AA)
            tip_x = attach_x + side * leaf_w * 2
            cv2.line(overlay, (attach_x, attach_y), (tip_x, attach_y), LEAF_VEIN, 1, cv2.LINE_AA)

        # --- Sunflower head: glow, two layers of petals, textured center ---
        tip_x, tip_y = self._tip_point()
        full_petals = int(self.petal_progress)
        partial_p = self.petal_progress - full_petals

        if full_petals > 0 or partial_p > 0:
            bloom_frac = self.petal_progress / self.max_petals

            # soft glow behind the head, grows with bloom progress
            base_glow_r = int(30 + 55 * bloom_frac)
            _soft_glow(overlay, (tip_x, tip_y), int(base_glow_r * 1.8), (10, 170, 255), 0.05)
            _soft_glow(overlay, (tip_x, tip_y), int(base_glow_r * 1.4), (15, 190, 255), 0.08)

            # outer petal layer
            for i in range(self.max_petals):
                growing = i < full_petals or (i == full_petals and partial_p > 0)
                if not growing:
                    continue
                size_frac = 1.0 if i < full_petals else max(0.15, partial_p)
                angle = 2 * math.pi * i / self.max_petals
                length = 46 * size_frac
                width = 20 * size_frac
                base = (tip_x, tip_y)
                tip_pt = (int(tip_x + length * math.cos(angle)), int(tip_y + length * math.sin(angle)))
                _draw_petal(overlay, base, tip_pt, width, PETAL_BASE, PETAL_TIP)

            # inner, slightly shorter petal layer, offset by half the angle step for a layered look
            half_step = math.pi / self.max_petals
            for i in range(self.max_petals):
                growing = i < full_petals or (i == full_petals and partial_p > 0)
                if not growing:
                    continue
                size_frac = 1.0 if i < full_petals else max(0.15, partial_p)
                angle = 2 * math.pi * i / self.max_petals + half_step
                length = 32 * size_frac
                width = 14 * size_frac
                base = (tip_x, tip_y)
                tip_pt = (int(tip_x + length * math.cos(angle)), int(tip_y + length * math.sin(angle)))
                _draw_petal(overlay, base, tip_pt, width, PETAL_BASE_INNER, PETAL_TIP_INNER)

            center_r = int(18 * min(1.0, self.petal_progress / 2))
            if center_r > 0:
                _draw_seed_center(overlay, (tip_x, tip_y), center_r)

        # Blend the drawing softly onto the real frame
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, dst=frame)
        return frame