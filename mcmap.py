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
    python mcmap.py ./world --error-log errors.txt  # list every unreadable chunk and why
    python mcmap.py ./world --debug-chunk 120 -340  # full dump of the chunk at block X=120 Z=-340

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
import traceback
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


def _lz4_block_decompress_py(src, dlen=None):
    """Pure-Python LZ4 block decoder, used when the 'lz4' package isn't installed."""
    dst = bytearray()
    i, n = 0, len(src)
    while i < n:
        token = src[i]
        i += 1
        lit = token >> 4
        if lit == 15:
            while True:
                b = src[i]
                i += 1
                lit += b
                if b != 255:
                    break
        dst += src[i:i + lit]
        i += lit
        if i >= n:
            break
        off = src[i] | (src[i + 1] << 8)
        i += 2
        ml = token & 15
        if ml == 15:
            while True:
                b = src[i]
                i += 1
                ml += b
                if b != 255:
                    break
        ml += 4
        if off == 0 or off > len(dst):
            raise ValueError("corrupt LZ4 block (bad match offset)")
        start = len(dst) - off
        if off >= ml:
            dst += dst[start:start + ml]
        else:  # overlapping copy = repeating pattern
            pat = bytes(dst[start:])
            dst += (pat * (ml // off + 1))[:ml]
    return bytes(dst)


try:
    import lz4.block as _lz4block

    def _lz4_block(chunk, dlen):
        return _lz4block.decompress(chunk, uncompressed_size=dlen)
except ImportError:
    _lz4block = None
    _lz4_block = _lz4_block_decompress_py


def _lz4_java_decompress(data):
    """Minecraft's LZ4 option uses lz4-java's LZ4BlockOutputStream framing."""
    out = bytearray()
    p = 0
    if data[:8] != b"LZ4Block":
        raise ValueError(f"LZ4 chunk without LZ4Block header (starts {bytes(data[:8])!r})")
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
            out += _lz4_block(bytes(chunk), dlen)
        else:
            raise ValueError(f"Unknown LZ4Block method 0x{method:02x}")
        if dlen == 0:
            break
    return bytes(out)


COMPRESSION_NAMES = {1: "gzip", 2: "zlib", 3: "none", 4: "lz4", 127: "custom"}


def _decompress(ctype, payload):
    if ctype == 2:
        return zlib.decompress(payload)
    if ctype == 1:
        return gzip.decompress(payload)
    if ctype == 3:
        return payload
    if ctype == 4:
        return _lz4_java_decompress(payload)
    if ctype == 127:  # payload starts with the algorithm's name as an NBT-style string
        try:
            name = bytes(payload[2:2 + _S_H.unpack_from(payload, 0)[0]]).decode("utf-8", "replace")
        except Exception:
            name = "?"
        raise ValueError(f"Custom compression algorithm '{name}' (set by server software)")
    raise ValueError(f"Unknown chunk compression type {ctype}")


class ChunkError(Exception):
    """A chunk that couldn't be read, with enough context to diagnose why."""

    def __init__(self, stage, message, ctype=None, detail=None):
        super().__init__(message)
        self.stage, self.message, self.ctype, self.detail = stage, message, ctype, detail


def locate_chunk(path, cx, cz, data=None):
    """Return (data, start, length, ctype, sectors) for a chunk in a region file, or None."""
    if data is None:
        with open(path, "rb") as f:
            data = f.read()
    i = (cx & 31) + (cz & 31) * 32
    o = i * 4
    sector = (data[o] << 16) | (data[o + 1] << 8) | data[o + 2]
    count = data[o + 3]
    if sector == 0 or count == 0:
        return None
    start = sector * 4096
    if start + 5 > len(data):
        return data, start, None, None, count
    return data, start, _S_i.unpack_from(data, start)[0], data[start + 4], count


def read_chunk_payload(path, data, cx, cz):
    """Return (ctype, raw_compressed_payload) or raise ChunkError."""
    loc = locate_chunk(path, cx, cz, data)
    if loc is None:
        return None
    _, start, length, ctype, sectors = loc
    if length is None:
        raise ChunkError("header", "chunk offset points past end of region file "
                                   "(truncated file - backup taken mid-write?)")
    if length <= 1 or start + 4 + length > len(data):
        raise ChunkError("header", f"chunk length {length} runs past end of region file "
                                   "(truncated file - backup taken mid-write?)", ctype & 0x7F)
    if length > sectors * 4096:
        raise ChunkError("header", f"chunk length {length} exceeds its {sectors} allocated "
                                   "sectors (corrupt header or file copied mid-write)", ctype & 0x7F)
    if ctype & 0x80:  # oversized chunk stored externally in c.X.Z.mcc
        ext = os.path.join(os.path.dirname(path), f"c.{cx}.{cz}.mcc")
        if not os.path.exists(ext):
            raise ChunkError("header", "oversized chunk's external c.X.Z.mcc file is missing",
                             ctype & 0x7F)
        with open(ext, "rb") as f:
            return ctype & 0x7F, f.read()
    return ctype, data[start + 5:start + 4 + length]


def iter_region_chunks(path, chunk_filter=None):
    """Yield (cx, cz, ctype, nbt_root_or_ChunkError) for every chunk in a region file."""
    m = _REGION_RE.match(os.path.basename(path))
    rx, rz = int(m.group(1)), int(m.group(2))
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 8192:
        if data:
            yield rx * 32, rz * 32, None, ChunkError(
                "header", f"region file is only {len(data)} bytes (needs 8192+ for its header)")
        return
    for i in range(1024):
        cx = rx * 32 + (i & 31)
        cz = rz * 32 + (i >> 5)
        if chunk_filter and not chunk_filter(cx, cz):
            continue
        try:
            got = read_chunk_payload(path, data, cx, cz)
        except ChunkError as e:
            yield cx, cz, e.ctype, e
            continue
        if got is None:
            continue
        ctype, payload = got
        try:
            raw = _decompress(ctype, payload)
        except Exception as e:
            yield cx, cz, ctype, ChunkError(
                "decompress", f"{COMPRESSION_NAMES.get(ctype, ctype)}: {type(e).__name__}: {e}",
                ctype)
            continue
        try:
            yield cx, cz, ctype, read_nbt(memoryview(raw))
        except Exception as e:
            yield cx, cz, ctype, ChunkError("nbt", f"{type(e).__name__}: {e}", ctype)


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

def unpack_indices(data, palette_len, spanning, count=4096, clamp=True):
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
    if not clamp:
        return vals[:count].astype(np.intp)
    return np.minimum(vals[:count], palette_len - 1).astype(np.intp)


def palette_name(p):
    """Block name from a palette entry, without namespace or block-state properties.

    Older versions store each entry as a compound ({Name: "minecraft:stone", Properties: {...}});
    newer versions (seen at DataVersion 5023 / 26.x) can store a plain string instead.
    Handle both, plus a string that carries inline state like "minecraft:oak_log[axis=y]".
    """
    if isinstance(p, dict):
        name = p.get("Name", "minecraft:air")
    elif isinstance(p, (str, bytes)):
        name = p.decode("utf-8", "replace") if isinstance(p, bytes) else p
    else:
        name = "minecraft:air"
    if name.lstrip().startswith("{"):  # SNBT compound written as a string
        m = re.search(r'Name\s*:\s*"?([A-Za-z0-9_.:/-]+)', name)
        name = m.group(1) if m else "minecraft:air"
    for sep in ("[", "{"):
        i = name.find(sep)
        if i >= 0:
            name = name[:i]
    return name.strip().split(":", 1)[-1] or "air"


def render_chunk(root, opts, unknown):
    """
    Returns (rgb[16,16,3] uint8, height[16,16] int16, valid[16,16] bool, water[16,16] bool)
    or a string explaining why the chunk was skipped (not generated, no blocks, ...).
    Arrays are indexed [z, x] within the chunk.
    """
    dv = root.get("DataVersion", 0)
    level = root["Level"] if "Level" in root else root
    status = str(level.get("Status", ""))
    if not opts["include_partial"] and not status.endswith("full"):
        return f"not fully generated (Status '{status or 'none'}')"

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
        names = [palette_name(p) for p in pal]
        if len(names) == 1 and names[0] in AIR:
            continue
        parsed[y] = (names, data)
    if not parsed:
        return "no block data"

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
            return "entirely above --ceiling"
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
    dbg = opts.get("_debug")
    if dbg is not None:
        by_id = list(local_ids)
        dbg["surface"] = Counter(by_id[i] for i in surf_c[valid].ravel())

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


_DIGITS = re.compile(r"\d+")


def _where(exc):
    """'func:line' of the innermost frame in this file, for render-stage errors."""
    frames = [f for f in traceback.extract_tb(exc.__traceback__)
              if os.path.basename(f.filename) == os.path.basename(__file__)]
    f = frames[-1] if frames else traceback.extract_tb(exc.__traceback__)[-1]
    return f"{f.name}:{f.lineno}"


class Stats:
    """Per-run diagnostics, merged across worker processes."""

    def __init__(self):
        self.compression = Counter()   # compression name -> chunks
        self.dv_ok = Counter()         # DataVersion -> chunks rendered
        self.dv_err = Counter()        # DataVersion -> chunks that failed after parsing
        self.skipped = Counter()       # reason -> chunks
        self.errors = Counter()        # (stage, normalized message) -> chunks
        self.samples = {}              # same key -> [(cx, cz, full message), ...]
        self.log = []                  # every failure, if --error-log was given

    def error(self, stage, msg, cx, cz, keep_log):
        key = (stage, _DIGITS.sub("#", msg))
        self.errors[key] += 1
        s = self.samples.setdefault(key, [])
        if len(s) < 3:
            s.append((cx, cz, msg))
        if keep_log:
            self.log.append(f"chunk {cx},{cz} (blocks {cx * 16},{cz * 16})\t{stage}\t{msg}")

    def merge(self, o):
        for a in ("compression", "dv_ok", "dv_err", "skipped", "errors"):
            getattr(self, a).update(getattr(o, a))
        for k, v in o.samples.items():
            s = self.samples.setdefault(k, [])
            s.extend(v[:3 - len(s)])
        self.log.extend(o.log)

    @property
    def error_count(self):
        return sum(self.errors.values())


def process_region(path, opts):
    bounds = opts["chunk_bounds"]
    keep_log = opts["keep_log"]
    flt = None
    if bounds:
        x0, z0, x1, z1 = bounds
        flt = lambda cx, cz: x0 <= cx <= x1 and z0 <= cz <= z1  # noqa: E731
    out, unknown, st = [], Counter(), Stats()
    try:
        for cx, cz, ctype, root in iter_region_chunks(path, flt):
            st.compression[COMPRESSION_NAMES.get(ctype, str(ctype))] += 1
            if isinstance(root, ChunkError):
                st.error(root.stage, root.message, cx, cz, keep_log)
                continue
            dv = root.get("DataVersion", 0) if isinstance(root, dict) else 0
            try:
                r = render_chunk(root, opts, unknown)
            except Exception as e:
                st.dv_err[dv] += 1
                st.error("render", f"{type(e).__name__} at {_where(e)}: {e} (DataVersion {dv})",
                         cx, cz, keep_log)
                continue
            if isinstance(r, str):
                st.skipped[r] += 1
            else:
                st.dv_ok[dv] += 1
                out.append((cx, cz) + r)
    except Exception as e:
        st.error("region", f"{os.path.basename(path)}: {type(e).__name__}: {e}", 0, 0, keep_log)
    return out, unknown, st


# Rough DataVersion -> release mapping, for the diagnostics report.
_DV_RELEASES = [(1519, "1.13"), (1952, "1.14"), (2225, "1.15"), (2566, "1.16"), (2724, "1.17"),
                (2860, "1.18"), (3105, "1.19"), (3463, "1.20"), (3953, "1.21+ / 26.x")]


def dv_label(dv):
    if not dv:
        return "no DataVersion"
    name = "pre-1.13"
    for v, n in _DV_RELEASES:
        if dv >= v:
            name = n
    return f"{dv} (~{name})"


def print_diagnostics(st, files_count, rendered):
    print("\n=== Diagnostics ===")
    print("Chunk compression:", ", ".join(f"{k}={v:,}" for k, v in st.compression.most_common()))
    if st.dv_ok:
        print("Rendered chunks by DataVersion:")
        for dv, n in sorted(st.dv_ok.items()):
            print(f"  {dv_label(dv):24s} {n:,}")
    if st.dv_err:
        print("Failed-to-render chunks by DataVersion:")
        for dv, n in sorted(st.dv_err.items()):
            print(f"  {dv_label(dv):24s} {n:,}")
    if st.skipped:
        print("Skipped (not errors):")
        for reason, n in st.skipped.most_common():
            print(f"  {n:>8,}  {reason}")
    if st.errors:
        print("Errors, grouped by cause:")
        for (stage, _), n in st.errors.most_common():
            key = (stage, _)
            print(f"  {n:>8,}  [{stage}] {st.samples[key][0][2]}")
            ex = "; ".join(f"chunk {cx},{cz} = block {cx * 16},{cz * 16}"
                           for cx, cz, _m in st.samples[key])
            print(f"            e.g. {ex}")
    print("Tip: run with --debug-chunk X Z (block coords from F3) on a failing spot for a full dump.")


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


def find_level_dat(start):
    """level.dat in the given folder or up to 5 parents (so a region folder works too)."""
    p = start
    for _ in range(6):
        cand = os.path.join(p, "level.dat")
        if os.path.isfile(cand):
            return cand
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return None


def read_markers(world, dim):
    """Spawn point and singleplayer player position from level.dat, if present."""
    markers = []
    path = find_level_dat(world)
    if not path:
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


def debug_chunk(region_dir, bx, bz, opts):
    """Print everything we can learn about one chunk and save its bytes for inspection."""
    cx, cz = bx >> 4, bz >> 4
    rx, rz = cx >> 5, cz >> 5
    path = os.path.join(region_dir, f"r.{rx}.{rz}.mca")
    print(f"Block {bx},{bz} -> chunk {cx},{cz} -> {path}")
    if not os.path.isfile(path):
        print("  Region file does not exist (never generated).")
        return
    print(f"  Region file size: {os.path.getsize(path):,} bytes")
    loc = locate_chunk(path, cx, cz)
    if loc is None:
        print("  Chunk is not present in the region file (never generated).")
        return
    data, start, length, ctype, sectors = loc
    print(f"  Header: offset {start:,}, {sectors} sector(s), length {length}, "
          f"compression byte {ctype} ({COMPRESSION_NAMES.get((ctype or 0) & 0x7F, '?')}"
          f"{', external .mcc' if ctype and ctype & 0x80 else ''})")
    try:
        ctype, payload = read_chunk_payload(path, data, cx, cz)
    except ChunkError as e:
        print(f"  FAILED reading payload: {e.message}")
        return
    base = f"chunk_{cx}_{cz}"
    print(f"  First bytes of payload: {bytes(payload[:16]).hex(' ')}")
    try:
        raw = _decompress(ctype, payload)
    except Exception:
        with open(base + ".compressed", "wb") as f:
            f.write(payload)
        print("  FAILED decompressing:")
        traceback.print_exc()
        print(f"  Saved compressed bytes to {base}.compressed")
        return
    with open(base + ".nbt", "wb") as f:
        f.write(raw)
    print(f"  Decompressed {len(raw):,} bytes -> saved to {base}.nbt")
    try:
        root = read_nbt(memoryview(raw))
    except Exception:
        print("  FAILED parsing NBT:")
        traceback.print_exc()
        return
    level = root["Level"] if "Level" in root else root
    print(f"  DataVersion: {dv_label(root.get('DataVersion', 0))}")
    print(f"  Layout: {'old (Level wrapper)' if 'Level' in root else 'new (1.18+)'}")
    print(f"  Status: {level.get('Status')!r}")
    print(f"  Top-level keys: {sorted(root.keys())}")
    if "Level" in root:
        print(f"  Level keys: {sorted(level.keys())}")
    secs = level.get("sections") or level.get("Sections") or []
    print(f"  Sections: {len(secs)}")

    def _pal(x):
        return (x.get("block_states") or {}).get("palette") or x.get("Palette") or []
    top_mixed = max((x.get("Y") for x in secs if len(_pal(x)) > 1), default=None)
    for s in secs:
        if "block_states" in s:
            pal, d = s["block_states"].get("palette"), s["block_states"].get("data")
        else:
            pal, d = s.get("Palette"), s.get("BlockStates")
        if not pal:
            print(f"    Y={s.get('Y')}: no palette (keys {sorted(s.keys())})")
            continue
        bits = max(4, (len(pal) - 1).bit_length())
        exp = 0 if len(pal) == 1 else (4096 + 64 // bits - 1) // (64 // bits)
        names = [palette_name(p) for p in pal]
        if pal and not isinstance(pal[0], dict):
            print(f"    (Y={s.get('Y')} palette entries are {type(pal[0]).__name__}, e.g. {pal[0]!r})")
        dlen = 0 if d is None else len(d)
        flag = "" if len(pal) == 1 or dlen in (exp, (4096 * bits + 63) // 64) else "  <-- unexpected length"
        if len(pal) > 1 and dlen:
            raw_idx = unpack_indices(d, len(pal), root.get("DataVersion", 0) < 2529, clamp=False)
            bad = int((raw_idx >= len(pal)).sum())
            if bad:
                flag += f"  <-- {bad} of 4096 indices exceed palette (max {int(raw_idx.max())})"
        print(f"    Y={s.get('Y'):>3}: palette {len(pal):>3}, data longs {dlen:>4} (expect {exp})"
              f"{flag}  e.g. {', '.join(names[:5])}")
        if len(pal) > 1 and s.get("Y") == top_mixed:
            print(f"      raw palette entries (first 8): {list(pal[:8])!r}")
    try:
        dopts = dict(opts, _debug={})
        r = render_chunk(root, dopts, Counter())
        print(f"  Render: {'skipped - ' + r if isinstance(r, str) else 'OK'}")
        surf = dopts["_debug"].get("surface")
        if surf:
            print("  Surface blocks the map shows for this chunk (of 256 columns):")
            for name, n in surf.most_common(15):
                rgb, known = block_color(name)
                tag = "" if known else "   <-- no color mapping, drawn as default gray"
                print(f"    {n:>4}  {name:30s} rgb{rgb}{tag}")
    except Exception:
        print("  Render FAILED:")
        traceback.print_exc()


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
    ap.add_argument("--diagnostics", action="store_true",
                    help="Always print the diagnostics report (it prints automatically on errors)")
    ap.add_argument("--error-log", metavar="FILE",
                    help="Write every unreadable chunk and the reason to this file")
    ap.add_argument("--debug-chunk", nargs=2, type=int, metavar=("X", "Z"),
                    help="Dump details for the chunk at block X Z (from F3) and exit")
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
        "keep_log": bool(args.error_log),
    }
    if args.debug_chunk:
        debug_chunk(region_dir, args.debug_chunk[0], args.debug_chunk[1], opts)
        return
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
    if _lz4block is None:
        print("(lz4 package not installed - LZ4 chunks will use the slower built-in decoder)")
    chunks, unknown, stats = [], Counter(), Stats()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(process_region, f, opts) for f in files]
        for n, fut in enumerate(as_completed(futs), 1):
            c, u, st = fut.result()
            chunks.extend(c)
            unknown.update(u)
            stats.merge(st)
            print(f"\r  {n}/{len(files)} regions, {len(chunks):,} chunks", end="", flush=True)
    print()
    errors = stats.error_count
    if args.error_log:
        with open(args.error_log, "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(stats.log)) + ("\n" if stats.log else ""))
        print(f"Wrote {len(stats.log):,} error line(s) to {args.error_log}")
    if errors or args.diagnostics or not chunks:
        print_diagnostics(stats, len(files), len(chunks))
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
