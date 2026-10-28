#!/usr/bin/env python3
"""
mcmap.py - Render a top-down map of the explored terrain in a Minecraft Java Edition world.

Requirements:  Python 3.9+, numpy, Pillow       (pip install numpy pillow)
Optional:      lz4  (only needed if the world/server uses LZ4 region compression)

Supports chunk formats from 1.13 through current (1.18+ "sections/block_states" layout,
the older "Level/Sections" layout, and the pre-1.16 packed-across-longs bit layout).

Examples:
    python mcmap.py "%APPDATA%\\.minecraft\\saves\\My World"
    python mcmap.py ~/minecraft/world -o nether.png --dimension nether
    python mcmap.py ./world --bounds -1000 -1000 1000 1000 --grid 256 --scale 0.5
    python mcmap.py ./world --list-unknown        # see which blocks fell back to the default color

The world can be open in-game while this runs (files are only read), but chunks Minecraft
hasn't flushed to disk yet won't appear. Close the world or run /save-all for a complete map.
"""

import argparse
import gzip
import os
import re
import struct
import sys
import time
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image, ImageDraw

# --------------------------------------------------------------------------------------
# NBT parsing
# --------------------------------------------------------------------------------------

_S_b = struct.Struct(">b")
_S_h = struct.Struct(">h")
_S_H = struct.Struct(">H")
_S_i = struct.Struct(">i")
_S_q = struct.Struct(">q")
_S_f = struct.Struct(">f")
_S_d = struct.Struct(">d")


class _NBT:
    """Minimal, fast-enough NBT reader. Returns dicts/lists/scalars; arrays become numpy."""

    __slots__ = ("b", "p")

    def __init__(self, buf):
        self.b = buf
        self.p = 0

    def root(self):
        if self.b[0] != 10:
            raise ValueError("NBT root is not a compound")
        self.p = 1
        self._str()  # root name (usually empty)
        return self._payload(10)

    def _str(self):
        n = _S_H.unpack_from(self.b, self.p)[0]
        self.p += 2
        s = bytes(self.b[self.p:self.p + n]).decode("utf-8", "replace")
        self.p += n
        return s

    def _payload(self, t):
        b = self.b
        p = self.p
        if t == 10:
            d = {}
            while True:
                tt = b[self.p]
                self.p += 1
                if tt == 0:
                    return d
                name = self._str()
                d[name] = self._payload(tt)
        if t == 1:
            self.p = p + 1
            return _S_b.unpack_from(b, p)[0]
        if t == 3:
            self.p = p + 4
            return _S_i.unpack_from(b, p)[0]
        if t == 8:
            return self._str()
        if t == 9:
            et = b[p]
            n = _S_i.unpack_from(b, p + 1)[0]
            self.p = p + 5
            return [self._payload(et) for _ in range(max(n, 0))]
        if t == 2:
            self.p = p + 2
            return _S_h.unpack_from(b, p)[0]
        if t == 4:
            self.p = p + 8
            return _S_q.unpack_from(b, p)[0]
        if t == 5:
            self.p = p + 4
            return _S_f.unpack_from(b, p)[0]
        if t == 6:
            self.p = p + 8
            return _S_d.unpack_from(b, p)[0]
        if t == 7:
            n = _S_i.unpack_from(b, p)[0]
            self.p = p + 4 + n
            return bytes(b[p + 4:p + 4 + n])
        if t == 11:
            n = _S_i.unpack_from(b, p)[0]
            self.p = p + 4 + 4 * n
            return np.frombuffer(b, dtype=">i4", count=n, offset=p + 4)
        if t == 12:
            n = _S_i.unpack_from(b, p)[0]
            self.p = p + 4 + 8 * n
            return np.frombuffer(b, dtype=">i8", count=n, offset=p + 4)
        raise ValueError(f"Unknown NBT tag type {t} at offset {p}")


def read_nbt(buf):
    return _NBT(buf).root()


# --------------------------------------------------------------------------------------
# Region (.mca) files
# --------------------------------------------------------------------------------------

_REGION_RE = re.compile(r"^r\.(-?\d+)\.(-?\d+)\.mca$")


def _lz4_java_decompress(data):
    """Minecraft's LZ4 option uses lz4-java's LZ4BlockOutputStream framing."""
    try:
        import lz4.block
    except ImportError:
        raise RuntimeError("Region uses LZ4 compression; run: pip install lz4")
    out = bytearray()
    p = 0
    while p + 21 <= len(data):
        if data[p:p + 8] != b"LZ4Block":
            break
        token = data[p + 8]
        clen, dlen = struct.unpack_from("<ii", data, p + 9)
        p += 21
        chunk = data[p:p + clen]
        p += clen
        method = token & 0xF0
        if method == 0x10:
            out += chunk
        elif method == 0x20:
            out += lz4.block.decompress(chunk, uncompressed_size=dlen)
        if dlen == 0:
            break
    return bytes(out)


def _decompress(ctype, payload):
    if ctype == 2:
        return zlib.decompress(payload)
    if ctype == 1:
        return gzip.decompress(payload)
    if ctype == 3:
        return payload
    if ctype == 4:
        return _lz4_java_decompress(payload)
    raise ValueError(f"Unknown chunk compression type {ctype}")


def iter_region_chunks(path, chunk_filter=None):
    """Yield (cx, cz, nbt_root) for every chunk stored in a region file."""
    m = _REGION_RE.match(os.path.basename(path))
    rx, rz = int(m.group(1)), int(m.group(2))
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 8192:
        return
    for i in range(1024):
        cx = rx * 32 + (i & 31)
        cz = rz * 32 + (i >> 5)
        if chunk_filter and not chunk_filter(cx, cz):
            continue
        o = i * 4
        sector = (data[o] << 16) | (data[o + 1] << 8) | data[o + 2]
        count = data[o + 3]
        if sector == 0 or count == 0:
            continue
        start = sector * 4096
        if start + 5 > len(data):
            continue
        length = _S_i.unpack_from(data, start)[0]
        ctype = data[start + 4]
        if ctype & 0x80:  # oversized chunk stored externally in c.X.Z.mcc
            ext = os.path.join(os.path.dirname(path), f"c.{cx}.{cz}.mcc")
            if not os.path.exists(ext):
                continue
            with open(ext, "rb") as f:
                payload = f.read()
            ctype &= 0x7F
        else:
            payload = data[start + 5:start + 4 + length]
        try:
            yield cx, cz, read_nbt(memoryview(_decompress(ctype, payload)))
        except Exception as e:  # corrupt chunk - skip it, don't kill the run
            yield cx, cz, e


# --------------------------------------------------------------------------------------
# Block colors
# --------------------------------------------------------------------------------------

# Things we look *through* when finding the surface.
AIR = {"air", "cave_air", "void_air", "barrier", "light", "structure_void"}
SEE_THROUGH = {
    "grass", "short_grass", "tall_grass", "fern", "large_fern", "dead_bush", "short_dry_grass",
    "tall_dry_grass", "bush", "firefly_bush", "leaf_litter", "wildflowers",
    "dandelion", "poppy", "blue_orchid", "allium", "azure_bluet", "oxeye_daisy", "cornflower",
    "lily_of_the_valley", "wither_rose", "sunflower", "lilac", "rose_bush", "peony",
    "torchflower", "pitcher_plant", "pink_petals", "closed_eyeblossom", "open_eyeblossom",
    "torch", "wall_torch", "soul_torch", "soul_wall_torch", "redstone_torch",
    "redstone_wall_torch", "redstone_wire", "tripwire", "tripwire_hook", "string", "lever",
    "rail", "powered_rail", "detector_rail", "activator_rail", "ladder", "vine", "cobweb",
    "sugar_cane", "sweet_berry_bush", "brown_mushroom", "red_mushroom", "nether_sprouts",
    "crimson_roots", "warped_roots", "glow_lichen", "sculk_vein", "hanging_roots",
    "spore_blossom", "cave_vines", "cave_vines_plant", "twisting_vines", "twisting_vines_plant",
    "weeping_vines", "weeping_vines_plant", "pale_hanging_moss", "small_dripleaf",
}
SEE_THROUGH_SUFFIXES = ("_sapling", "_tulip", "_button", "_pressure_plate", "_sign",
                        "_wall_sign", "_banner", "_wall_banner", "_head", "_skull",
                        "_coral_fan", "_coral_wall_fan", "_candle", "candle")
# Water and things that only live in water: drawn as water, depth measured through them.
WATERY = {"water", "bubble_column", "seagrass", "tall_seagrass", "kelp", "kelp_plant",
          "sea_pickle"}

WATER_RGB = (52, 87, 186)

COLORS = {
    # natural terrain
    "grass_block": (110, 158, 66), "dirt": (134, 96, 67), "coarse_dirt": (119, 85, 59),
    "rooted_dirt": (144, 103, 76), "podzol": (91, 63, 24), "mycelium": (111, 99, 105),
    "dirt_path": (148, 122, 65), "farmland": (110, 73, 45), "mud": (60, 57, 60),
    "moss_block": (89, 110, 45), "pale_moss_block": (106, 112, 102), "clay": (160, 166, 179),
    "gravel": (131, 127, 126), "sand": (219, 207, 163), "red_sand": (190, 102, 33),
    "sandstone": (216, 203, 155), "red_sandstone": (186, 99, 29),
    "stone": (125, 125, 125), "cobblestone": (122, 122, 122), "mossy_cobblestone": (110, 118, 94),
    "granite": (149, 103, 85), "diorite": (188, 188, 188), "andesite": (136, 136, 136),
    "tuff": (108, 109, 102), "calcite": (223, 224, 220), "deepslate": (80, 80, 82),
    "cobbled_deepslate": (77, 77, 80), "bedrock": (85, 85, 85), "obsidian": (21, 18, 30),
    "snow": (249, 254, 254), "snow_block": (249, 254, 254), "powder_snow": (248, 253, 253),
    "ice": (145, 183, 253), "packed_ice": (141, 180, 250), "blue_ice": (116, 167, 253),
    "lava": (207, 92, 15), "magma_block": (142, 63, 31),
    "pointed_dripstone": (129, 102, 89), "dripstone_block": (134, 107, 92),
    "amethyst_block": (133, 97, 191), "budding_amethyst": (132, 96, 186),
    "lily_pad": (32, 128, 48), "cactus": (85, 127, 43), "pumpkin": (198, 118, 24),
    "melon": (111, 145, 30), "hay_block": (166, 136, 38), "bamboo": (94, 144, 28),
    "big_dripleaf": (112, 142, 52), "azalea": (101, 124, 47), "flowering_azalea": (112, 121, 82),
    "mangrove_roots": (74, 59, 38), "muddy_mangrove_roots": (70, 59, 45),
    "brown_mushroom_block": (149, 111, 81), "red_mushroom_block": (200, 46, 45),
    "mushroom_stem": (203, 196, 185), "terracotta": (152, 94, 67), "suspicious_sand": (217, 204, 159),
    # leaves (untinted approximations of typical biome tints)
    "oak_leaves": (59, 110, 33), "birch_leaves": (96, 131, 61), "spruce_leaves": (55, 90, 55),
    "jungle_leaves": (48, 120, 22), "acacia_leaves": (88, 120, 30), "dark_oak_leaves": (48, 98, 24),
    "mangrove_leaves": (68, 120, 36), "cherry_leaves": (229, 173, 194),
    "azalea_leaves": (90, 115, 44), "flowering_azalea_leaves": (100, 111, 61),
    "pale_oak_leaves": (138, 145, 136),
    # nether
    "netherrack": (97, 38, 38), "crimson_nylium": (130, 31, 31), "warped_nylium": (43, 114, 101),
    "soul_sand": (81, 62, 50), "soul_soil": (75, 57, 46), "basalt": (80, 81, 86),
    "blackstone": (42, 36, 41), "glowstone": (171, 131, 84), "nether_wart_block": (114, 2, 2),
    "warped_wart_block": (22, 119, 121), "shroomlight": (240, 146, 70), "nether_gold_ore": (115, 54, 42),
    "ancient_debris": (94, 66, 58), "nether_quartz_ore": (117, 65, 62), "crying_obsidian": (32, 10, 60),
    # end
    "end_stone": (219, 222, 158), "purpur_block": (169, 125, 169), "end_stone_bricks": (218, 224, 162),
    "chorus_plant": (93, 57, 93), "chorus_flower": (151, 120, 151),
    # common builds
    "glass": (200, 220, 230), "bricks": (150, 97, 83), "stone_bricks": (122, 121, 122),
    "smooth_stone": (159, 159, 159), "quartz_block": (235, 229, 222), "bookshelf": (117, 94, 59),
    "crafting_table": (120, 73, 42), "furnace": (110, 110, 110), "chest": (162, 113, 47),
    "barrel": (135, 102, 58), "iron_block": (220, 220, 220), "gold_block": (246, 208, 61),
    "diamond_block": (98, 237, 228), "emerald_block": (42, 203, 87), "lapis_block": (30, 67, 140),
    "redstone_block": (175, 24, 5), "coal_block": (16, 15, 15), "copper_block": (192, 107, 79),
    "prismarine": (99, 156, 151), "sponge": (195, 192, 74), "tnt": (182, 63, 50),
    "slime_block": (111, 192, 91), "honey_block": (251, 185, 52), "target": (229, 176, 168),
    "sculk": (12, 29, 36), "reinforced_deepslate": (80, 82, 78),
}

DYE = {
    "white": (233, 236, 236), "orange": (240, 118, 19), "magenta": (189, 68, 179),
    "light_blue": (58, 175, 217), "yellow": (248, 197, 39), "lime": (112, 185, 25),
    "pink": (237, 141, 172), "gray": (62, 68, 71), "light_gray": (142, 142, 134),
    "cyan": (21, 137, 145), "purple": (121, 42, 172), "blue": (53, 57, 157),
    "brown": (114, 71, 40), "green": (84, 109, 27), "red": (160, 39, 34), "black": (20, 21, 25),
}
_DYE_ORDER = sorted(DYE, key=len, reverse=True)  # match "light_blue" before "blue"

WOOD = {"oak": (162, 130, 78), "spruce": (114, 84, 48), "birch": (192, 175, 121),
        "jungle": (160, 115, 80), "acacia": (168, 90, 50), "dark_oak": (66, 43, 20),
        "mangrove": (117, 54, 48), "cherry": (226, 178, 172), "pale_oak": (227, 217, 214),
        "bamboo": (193, 173, 80), "crimson": (101, 48, 70), "warped": (43, 104, 99)}
_WOOD_ORDER = sorted(WOOD, key=len, reverse=True)

# Substring fallbacks, checked in order.
_FALLBACKS = [
    ("water", WATER_RGB), ("lava", (207, 92, 15)), ("leaves", (59, 110, 33)),
    ("deepslate", (80, 80, 82)), ("blackstone", (42, 36, 41)), ("netherrack", (97, 38, 38)),
    ("nether_brick", (44, 22, 26)), ("end_stone", (219, 222, 158)), ("purpur", (169, 125, 169)),
    ("prismarine", (99, 156, 151)), ("red_sandstone", (186, 99, 29)), ("sandstone", (216, 203, 155)),
    ("red_sand", (190, 102, 33)), ("sand", (219, 207, 163)), ("snow", (249, 254, 254)),
    ("ice", (145, 183, 253)), ("copper", (192, 107, 79)), ("quartz", (235, 229, 222)),
    ("brick", (150, 97, 83)), ("tuff", (108, 109, 102)), ("granite", (149, 103, 85)),
    ("diorite", (188, 188, 188)), ("andesite", (136, 136, 136)), ("cobblestone", (122, 122, 122)),
    ("stone", (125, 125, 125)), ("_ore", (120, 120, 120)), ("glass", (200, 220, 230)),
    ("coral", (180, 80, 120)), ("mud", (60, 57, 60)), ("moss", (89, 110, 45)),
    ("iron", (200, 200, 200)), ("gold", (246, 208, 61)), ("amethyst", (133, 97, 191)),
]
DEFAULT_RGB = (128, 128, 128)

_color_cache = {}


def block_color(name):
    """Return (rgb, known) for a block name without the 'minecraft:' prefix."""
    hit = _color_cache.get(name)
    if hit is not None:
        return hit
    rgb, known = DEFAULT_RGB, True
    if name in COLORS:
        rgb = COLORS[name]
    else:
        dye = next((d for d in _DYE_ORDER if name.startswith(d + "_")), None)
        wood = next((w for w in _WOOD_ORDER if name.startswith(w + "_")), None)
        if dye and any(k in name for k in ("wool", "concrete", "terracotta", "carpet", "glass",
                                           "bed", "shulker_box", "candle", "banner")):
            rgb = DYE[dye]
            if "terracotta" in name and "glazed" not in name:
                rgb = tuple(int(c * 0.6 + t * 0.4) for c, t in zip(rgb, (152, 94, 67)))
        elif wood and any(k in name for k in ("log", "wood", "stem", "hyphae")):
            rgb = (102, 81, 51) if wood not in ("crimson", "warped", "cherry", "pale_oak", "birch") \
                else tuple(int(c * 0.8) for c in WOOD[wood])
        elif wood:  # planks, slabs, stairs, fences, doors, roofs...
            rgb = WOOD[wood]
        else:
            for key, c in _FALLBACKS:
                if key in name:
                    rgb = c
                    break
            else:
                known = False
    _color_cache[name] = (rgb, known)
    return rgb, known


def classify(name):
    """0 = look through, 1 = water-ish, 2 = solid surface."""
    if name in AIR or name in SEE_THROUGH or name.endswith(SEE_THROUGH_SUFFIXES):
        return 0
    if name in WATERY:
        return 1
    return 2


# --------------------------------------------------------------------------------------
# Chunk decoding
# --------------------------------------------------------------------------------------

def unpack_indices(data, palette_len, spanning, count=4096):
    """Decode a packed long array of palette indices."""
    if data is None or len(data) == 0 or palette_len <= 1:
        return np.zeros(count, dtype=np.intp)
    bits = max(4, (palette_len - 1).bit_length())
    longs = np.asarray(data).astype(np.int64).view(np.uint64)
    mask = np.uint64((1 << bits) - 1)
    if not spanning:  # 1.16+: values never straddle two longs
        per = 64 // bits
        shifts = np.arange(per, dtype=np.uint64) * np.uint64(bits)
        vals = ((longs[:, None] >> shifts[None, :]) & mask).reshape(-1)
    else:  # 1.13 - 1.15: one continuous bit stream
        pos = np.arange(count, dtype=np.int64) * bits
        li = np.minimum(pos >> 6, len(longs) - 1)
        sh = (pos & 63).astype(np.uint64)
        vals = longs[li] >> sh
        over = (pos & 63) + bits > 64
        if over.any():
            nxt = longs[np.minimum(li[over] + 1, len(longs) - 1)]
            vals[over] |= nxt << (np.uint64(64) - sh[over])
        vals &= mask
    if vals.size < count:
        vals = np.concatenate([vals, np.zeros(count - vals.size, dtype=np.uint64)])
    return np.minimum(vals[:count], palette_len - 1).astype(np.intp)


def render_chunk(root, opts, unknown):
    """
    Returns (rgb[16,16,3] uint8, height[16,16] int16, valid[16,16] bool, water[16,16] bool)
    or None if the chunk isn't fully generated / has no blocks.
    Arrays are indexed [z, x] within the chunk.
    """
    dv = root.get("DataVersion", 0)
    level = root["Level"] if "Level" in root else root
    status = str(level.get("Status", ""))
    if not opts["include_partial"] and not status.endswith("full"):
        return None

    sections = level.get("sections") or level.get("Sections") or []
    spanning = dv < 2529  # before 20w17a (1.16), values could straddle longs
    parsed = {}
    for s in sections:
        y = s.get("Y")
        if y is None:
            continue
        if "block_states" in s:
            pal = s["block_states"].get("palette")
            data = s["block_states"].get("data")
        else:
            pal, data = s.get("Palette"), s.get("BlockStates")
        if not pal:
            continue
        names = [p.get("Name", "minecraft:air").split(":", 1)[-1] for p in pal]
        if len(names) == 1 and names[0] in AIR:
            continue
        parsed[y] = (names, data)
    if not parsed:
        return None

    lo, hi = min(parsed), max(parsed)
    H = (hi - lo + 1) * 16
    kind = np.zeros((H, 16, 16), dtype=np.uint8)
    cid = np.zeros((H, 16, 16), dtype=np.uint16)
    local_ids, colors = {}, []

    for y, (names, data) in parsed.items():
        ids = np.empty(len(names), dtype=np.uint16)
        kinds = np.empty(len(names), dtype=np.uint8)
        for i, n in enumerate(names):
            j = local_ids.get(n)
            if j is None:
                j = local_ids[n] = len(colors)
                rgb, known = block_color(n)
                colors.append(rgb)
                if not known and classify(n) == 2:
                    unknown[n] += 1
            ids[i] = j
            kinds[i] = classify(n)
        idx = unpack_indices(data, len(names), spanning).reshape(16, 16, 16)  # [y, z, x]
        r = (y - lo) * 16
        kind[r:r + 16] = kinds[idx]
        cid[r:r + 16] = ids[idx]

    top_world_y = lo * 16 + H - 1
    # Flip so row 0 is the highest block, then apply a ceiling if requested.
    K = kind[::-1]
    C = cid[::-1]
    if opts["ceiling"] is not None:
        cut = top_world_y - opts["ceiling"]
        if cut >= H:
            return None
        if cut > 0:
            K, C = K[cut:], C[cut:]
            top_world_y -= cut

    look_through = K == 0
    candidate = ~look_through
    if opts["need_air_above"]:  # e.g. the Nether: skip the bedrock roof
        candidate &= np.maximum.accumulate(look_through, axis=0)
    valid = candidate.any(axis=0)
    first = candidate.argmax(axis=0)  # [z, x]
    zz, xx = np.indices((16, 16))

    height = (top_world_y - first).astype(np.int16)
    surf_kind = K[first, zz, xx]
    surf_c = C[first, zz, xx]
    palette_rgb = np.array(colors, dtype=np.float32)
    rgb = palette_rgb[surf_c]

    water = valid & (surf_kind == 1)
    if water.any():
        rows = np.arange(K.shape[0])[:, None, None]
        floor_mask = (K == 2) & (rows > first[None])
        has_floor = floor_mask.any(axis=0)
        floor = floor_mask.argmax(axis=0)
        depth = np.where(has_floor, floor - first, 64).astype(np.float32)
        floor_rgb = palette_rgb[C[floor, zz, xx]]
        # Shallow water shows the bottom; deep water goes dark blue.
        a = np.clip(0.55 + depth * 0.05, 0.55, 1.0)[..., None]
        wrgb = np.array(WATER_RGB, dtype=np.float32) * a + floor_rgb * (1 - a)
        dark = np.clip(1.08 - depth * 0.012, 0.6, 1.08)[..., None]
        rgb = np.where(water[..., None], wrgb * dark, rgb)

    rgb[~valid] = 0
    return np.clip(rgb, 0, 255).astype(np.uint8), height, valid, water


def process_region(path, opts):
    bounds = opts["chunk_bounds"]
    flt = None
    if bounds:
        x0, z0, x1, z1 = bounds
        flt = lambda cx, cz: x0 <= cx <= x1 and z0 <= cz <= z1  # noqa: E731
    out, unknown, errors = [], Counter(), 0
    try:
        for cx, cz, root in iter_region_chunks(path, flt):
            if isinstance(root, Exception):
                errors += 1
                continue
            try:
                r = render_chunk(root, opts, unknown)
            except Exception:
                errors += 1
                continue
            if r is not None:
                out.append((cx, cz) + r)
    except Exception:
        errors += 1
    return out, unknown, errors


# --------------------------------------------------------------------------------------
# World-level helpers
# --------------------------------------------------------------------------------------

DIM_DIRS = {
    "overworld": ["region", "dimensions/minecraft/overworld/region"],
    "nether": ["DIM-1/region", "dimensions/minecraft/the_nether/region"],
    "end": ["DIM1/region", "dimensions/minecraft/the_end/region"],
}
DIM_IDS = {"overworld": ("minecraft:overworld", 0), "nether": ("minecraft:the_nether", -1),
           "end": ("minecraft:the_end", 1)}


def find_region_dir(world, dim):
    if any(_REGION_RE.match(f) for f in os.listdir(world)):
        return world  # pointed straight at a region folder
    for rel in DIM_DIRS[dim]:
        p = os.path.join(world, rel)
        if os.path.isdir(p):
            return p
    return None


def read_markers(world, dim):
    """Spawn point and singleplayer player position from level.dat, if present."""
    markers = []
    path = os.path.join(world, "level.dat")
    if not os.path.isfile(path):
        return markers
    try:
        with gzip.open(path, "rb") as f:
            data = read_nbt(memoryview(f.read())).get("Data", {})
    except Exception:
        return markers
    if dim == "overworld":
        if "SpawnX" in data:
            markers.append(("Spawn", data["SpawnX"], data["SpawnZ"], (220, 40, 40)))
        elif isinstance(data.get("spawn"), dict) and data["spawn"].get("pos") is not None:
            pos = data["spawn"]["pos"]
            markers.append(("Spawn", int(pos[0]), int(pos[2]), (220, 40, 40)))
    player = data.get("Player")
    if isinstance(player, dict) and player.get("Pos"):
        pdim = player.get("Dimension", "minecraft:overworld")
        if pdim in DIM_IDS[dim]:
            pos = player["Pos"]
            markers.append(("Player", int(pos[0] // 1), int(pos[2] // 1), (40, 90, 230)))
    return markers


def parse_color(s):
    s = s.lstrip("#")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4)) + (255,)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Render a map of the explored terrain in a "
                                             "Minecraft Java Edition world.")
    ap.add_argument("world", help="World folder (the one containing level.dat), or a region folder")
    ap.add_argument("-o", "--output", default=None, help="Output PNG (default: <world>_<dim>.png)")
    ap.add_argument("-d", "--dimension", choices=["overworld", "nether", "end"], default="overworld")
    ap.add_argument("--bounds", nargs=4, type=int, metavar=("X1", "Z1", "X2", "Z2"),
                    help="Only render this block-coordinate rectangle")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Pixels per block (e.g. 0.5 to halve the size, 2 to double)")
    ap.add_argument("--grid", type=int, default=0, metavar="N",
                    help="Draw a labeled grid every N blocks (e.g. 512)")
    ap.add_argument("--ceiling", type=int, default=None,
                    help="Ignore blocks above this Y (default: 120 in the Nether)")
    ap.add_argument("--no-shade", action="store_true", help="Disable hill shading")
    ap.add_argument("--no-markers", action="store_true", help="Don't draw spawn/player markers")
    ap.add_argument("--background", default=None, metavar="#RRGGBB",
                    help="Fill unexplored area with a color instead of transparency")
    ap.add_argument("--include-partial", action="store_true",
                    help="Include chunks that are only partially generated (beyond explored edge)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--list-unknown", action="store_true",
                    help="Print blocks that had no color mapping (so you can add them)")
    args = ap.parse_args()

    world = os.path.abspath(os.path.expanduser(os.path.expandvars(args.world)))
    if not os.path.isdir(world):
        sys.exit(f"Not a folder: {world}")
    region_dir = find_region_dir(world, args.dimension)
    if not region_dir:
        sys.exit(f"No region folder found for the {args.dimension} in {world}")

    nether = args.dimension == "nether"
    opts = {
        "include_partial": args.include_partial,
        "ceiling": args.ceiling if args.ceiling is not None else (120 if nether else None),
        "need_air_above": nether,
        "chunk_bounds": None,
    }
    if args.bounds:
        x1, z1, x2, z2 = args.bounds
        x1, x2 = sorted((x1, x2))
        z1, z2 = sorted((z1, z2))
        opts["chunk_bounds"] = (x1 >> 4, z1 >> 4, x2 >> 4, z2 >> 4)

    files = []
    for f in os.listdir(region_dir):
        m = _REGION_RE.match(f)
        if not m:
            continue
        if opts["chunk_bounds"]:
            rx, rz = int(m.group(1)), int(m.group(2))
            cx0, cz0, cx1, cz1 = opts["chunk_bounds"]
            if rx * 32 + 31 < cx0 or rx * 32 > cx1 or rz * 32 + 31 < cz0 or rz * 32 > cz1:
                continue
        files.append(os.path.join(region_dir, f))
    if not files:
        sys.exit("No region files to render.")

    t0 = time.time()
    print(f"Reading {len(files)} region file(s) from {region_dir} using {args.workers} worker(s)...")
    chunks, unknown, errors = [], Counter(), 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(process_region, f, opts) for f in files]
        for n, fut in enumerate(as_completed(futs), 1):
            c, u, e = fut.result()
            chunks.extend(c)
            unknown.update(u)
            errors += e
            print(f"\r  {n}/{len(files)} regions, {len(chunks):,} chunks", end="", flush=True)
    print()
    if not chunks:
        sys.exit("No explored chunks found (try --include-partial).")

    # ---- assemble ----
    min_cx = min(c[0] for c in chunks)
    min_cz = min(c[1] for c in chunks)
    max_cx = max(c[0] for c in chunks)
    max_cz = max(c[1] for c in chunks)
    W, Hh = (max_cx - min_cx + 1) * 16, (max_cz - min_cz + 1) * 16
    px = W * Hh
    if px > 600_000_000:
        sys.exit(f"Map would be {W}x{Hh} pixels; use --bounds to render a smaller area.")
    rgb = np.zeros((Hh, W, 3), dtype=np.uint8)
    height = np.zeros((Hh, W), dtype=np.int16)
    valid = np.zeros((Hh, W), dtype=bool)
    water = np.zeros((Hh, W), dtype=bool)
    for cx, cz, c_rgb, c_h, c_v, c_w in chunks:
        x, z = (cx - min_cx) * 16, (cz - min_cz) * 16
        rgb[z:z + 16, x:x + 16] = c_rgb
        height[z:z + 16, x:x + 16] = c_h
        valid[z:z + 16, x:x + 16] = c_v
        water[z:z + 16, x:x + 16] = c_w
    del chunks
    origin_x, origin_z = min_cx * 16, min_cz * 16

    # ---- hill shading: light from the north-west, like in-game maps ----
    if not args.no_shade:
        h = height.astype(np.float32)
        nw = h.copy()
        nw[1:, 1:] = h[:-1, :-1]
        nv = valid.copy()
        nv[1:, 1:] = valid[:-1, :-1]
        dh = np.where(valid & nv, h - nw, 0)
        f = np.clip(1.0 + dh * 0.06, 0.65, 1.3)
        f[water] = 1.0
        rgb = np.clip(rgb.astype(np.float32) * f[..., None], 0, 255).astype(np.uint8)
        del h, nw, nv, dh, f

    # ---- crop to requested bounds ----
    if args.bounds:
        bx1, bz1 = max(x1 - origin_x, 0), max(z1 - origin_z, 0)
        bx2, bz2 = min(x2 - origin_x + 1, W), min(z2 - origin_z + 1, Hh)
        rgb, valid = rgb[bz1:bz2, bx1:bx2], valid[bz1:bz2, bx1:bx2]
        origin_x += bx1
        origin_z += bz1

    alpha = np.where(valid, 255, 0).astype(np.uint8)
    img = Image.fromarray(np.dstack([rgb, alpha]), "RGBA")
    del rgb, valid, alpha, height, water

    if args.scale != 1.0:
        size = (max(1, round(img.width * args.scale)), max(1, round(img.height * args.scale)))
        img = img.resize(size, Image.NEAREST if args.scale > 1 else Image.BOX)
    s = args.scale

    if args.background:
        bg = Image.new("RGBA", img.size, parse_color(args.background))
        img = Image.alpha_composite(bg, img)

    def to_px(bx, bz):
        return (bx - origin_x) * s, (bz - origin_z) * s

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    if args.grid and args.grid > 0:
        g = args.grid
        x_world_max = origin_x + img.width / s
        z_world_max = origin_z + img.height / s
        gx = (origin_x // g + 1) * g if origin_x % g else origin_x
        while gx < x_world_max:
            X, _ = to_px(gx, 0)
            draw.line([(X, 0), (X, img.height)], fill=(0, 0, 0, 90), width=1)
            draw.text((X + 3, 3), f"x={gx}", fill=(0, 0, 0, 200))
            gx += g
        gz = (origin_z // g + 1) * g if origin_z % g else origin_z
        while gz < z_world_max:
            _, Z = to_px(0, gz)
            draw.line([(0, Z), (img.width, Z)], fill=(0, 0, 0, 90), width=1)
            draw.text((3, Z + 3), f"z={gz}", fill=(0, 0, 0, 200))
            gz += g
    if not args.no_markers:
        for label, mx, mz, color in read_markers(world, args.dimension):
            X, Z = to_px(mx + 0.5, mz + 0.5)
            if not (0 <= X < img.width and 0 <= Z < img.height):
                continue
            r = max(4, int(5 * min(s, 2)))
            draw.ellipse([X - r, Z - r, X + r, Z + r], fill=color + (255,), outline=(255, 255, 255, 255),
                         width=2)
            draw.text((X + r + 3, Z - 6), f"{label} ({mx}, {mz})", fill=(255, 255, 255, 255),
                      stroke_width=2, stroke_fill=(0, 0, 0, 255))
    img = Image.alpha_composite(img, overlay)

    out = args.output or f"{os.path.basename(world.rstrip(os.sep))}_{args.dimension}.png"
    img.save(out, optimize=False)
    print(f"Saved {out}  ({img.width}x{img.height} px, {time.time() - t0:.1f}s)")
    print(f"Top-left pixel = block X {origin_x}, Z {origin_z}; {s:g} pixel(s) per block.")
    if errors:
        print(f"Skipped {errors} unreadable chunk(s).")
    if args.list_unknown and unknown:
        print("\nBlocks drawn with the default gray (add them to COLORS):")
        for name, n in unknown.most_common(40):
            print(f"  {name:40s} {n}")


if __name__ == "__main__":
    main()
