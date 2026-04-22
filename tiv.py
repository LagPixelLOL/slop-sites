import argparse
import os
import sys
from pathlib import Path
from functools import lru_cache

import numpy as np
from PIL import Image


FLAG_FG = 1
FLAG_BG = 2
FLAG_MODE_256 = 4
FLAG_NOOPT = 16

COLOR_STEPS = (0, 0x5f, 0x87, 0xaf, 0xd7, 0xff)
GRAYSCALE_STEPS = (
    0x08, 0x12, 0x1c, 0x26, 0x30, 0x3a, 0x44, 0x4e,
    0x58, 0x62, 0x6c, 0x76, 0x80, 0x8a, 0x94, 0x9e,
    0xa8, 0xb2, 0xbc, 0xc6, 0xd0, 0xda, 0xe4, 0xee,
)

BITMAPS = [
    (0x00000000, 0x00a0),
    (0x0000000f, 0x2581),
    (0x000000ff, 0x2582),
    (0x00000fff, 0x2583),
    (0x0000ffff, 0x2584),
    (0x000fffff, 0x2585),
    (0x00ffffff, 0x2586),
    (0x0fffffff, 0x2587),
    (0xeeeeeeee, 0x258a),
    (0xcccccccc, 0x258c),
    (0x88888888, 0x258e),
    (0x0000cccc, 0x2596),
    (0x00003333, 0x2597),
    (0xcccc0000, 0x2598),
    (0xcccc3333, 0x259a),
    (0x33330000, 0x259d),
    (0x000ff000, 0x2501),
    (0x66666666, 0x2503),
    (0x00077666, 0x250f),
    (0x000ee666, 0x2513),
    (0x66677000, 0x2517),
    (0x666ee000, 0x251b),
    (0x66677666, 0x2523),
    (0x666ee666, 0x252b),
    (0x000ff666, 0x2533),
    (0x666ff000, 0x253b),
    (0x666ff666, 0x254b),
    (0x000cc000, 0x2578),
    (0x00066000, 0x2579),
    (0x00033000, 0x257a),
    (0x00066000, 0x257b),
    (0x06600660, 0x254f),
    (0x000f0000, 0x2500),
    (0x0000f000, 0x2500),
    (0x44444444, 0x2502),
    (0x22222222, 0x2502),
    (0x000e0000, 0x2574),
    (0x0000e000, 0x2574),
    (0x44440000, 0x2575),
    (0x22220000, 0x2575),
    (0x00030000, 0x2576),
    (0x00003000, 0x2576),
    (0x00004444, 0x2577),
    (0x00002222, 0x2577),
    (0x44444444, 0x23a2),
    (0x22222222, 0x23a5),
    (0x0f000000, 0x23ba),
    (0x00f00000, 0x23bb),
    (0x00000f00, 0x23bc),
    (0x000000f0, 0x23bd),
    (0x00066000, 0x25aa),
    (0x000137f0, 0x25e2),
    (0x0008cef0, 0x25e3),
    (0x000fec80, 0x25e4),
    (0x000f7310, 0x25e5),
]


def _build_step_lookup(steps):
    table = [0] * 256
    for v in range(256):
        best = 0
        best_diff = abs(steps[0] - v)
        for i in range(1, len(steps)):
            d = abs(steps[i] - v)
            if d < best_diff:
                best_diff = d
                best = i
        table[v] = best
    return tuple(table)


_COLOR_STEP_LOOKUP = _build_step_lookup(COLOR_STEPS)
_GRAY_STEP_LOOKUP = _build_step_lookup(GRAYSCALE_STEPS)


@lru_cache(maxsize=256)
def _pattern_mask(pattern):
    bits = [(pattern >> (31 - i)) & 1 for i in range(32)]
    return np.array(bits, dtype=bool).reshape(8, 4)


class CharData:
    __slots__ = ("fg", "bg", "codepoint")

    def __init__(self, fg, bg, codepoint):
        self.fg = fg
        self.bg = bg
        self.codepoint = codepoint


def _avg(pixels):
    if len(pixels) == 0:
        return (0, 0, 0)
    m = pixels.mean(axis=0)
    return (int(m[0]), int(m[1]), int(m[2]))


def create_char_data(block, codepoint, pattern):
    mask = _pattern_mask(pattern)
    return CharData(_avg(block[mask]), _avg(block[~mask]), codepoint)


def find_char_data(block, flags):
    flat = block.reshape(-1, 3).astype(np.int32)

    packed_arr = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]
    counts = {}
    for c in packed_arr.tolist():
        counts[c] = counts.get(c, 0) + 1

    items = sorted(counts.items(), key=lambda kv: -kv[1])
    c1, cnt1 = items[0]
    if len(items) > 1:
        c2, cnt2 = items[1]
        total = cnt1 + cnt2
    else:
        c2, total = c1, cnt1

    direct = total > 16

    if direct:
        c1_rgb = np.array(
            [(c1 >> 16) & 255, (c1 >> 8) & 255, c1 & 255], dtype=np.int32)
        c2_rgb = np.array(
            [(c2 >> 16) & 255, (c2 >> 8) & 255, c2 & 255], dtype=np.int32)
        d1 = flat - c1_rgb
        d2 = flat - c2_rgb
        bits_arr = (d1 * d1).sum(axis=1) > (d2 * d2).sum(axis=1)
    else:
        min_c = flat.min(axis=0)
        max_c = flat.max(axis=0)
        ranges = max_c - min_c
        split_idx = int(np.argmax(ranges))
        split_val = int(min_c[split_idx]) + int(ranges[split_idx]) // 2
        bits_arr = flat[:, split_idx] > split_val

    bits = int.from_bytes(np.packbits(bits_arr).tobytes(), "big")

    best_diff = 8
    best_pattern = 0x0000ffff
    codepoint = 0x2584
    inverted = False

    for pat, cp in BITMAPS:
        diff = ((pat ^ bits) & 0xffffffff).bit_count()
        if diff < best_diff:
            best_pattern = pat
            codepoint = cp
            best_diff = diff
            inverted = False
        inv_diff = 32 - diff
        if inv_diff < best_diff:
            best_pattern = pat
            codepoint = cp
            best_diff = inv_diff
            inverted = True

    if direct:
        if inverted:
            c1, c2 = c2, c1
        fg = ((c2 >> 16) & 255, (c2 >> 8) & 255, c2 & 255)
        bg = ((c1 >> 16) & 255, (c1 >> 8) & 255, c1 & 255)
        return CharData(fg, bg, codepoint)

    return create_char_data(block, codepoint, best_pattern)


def clamp_byte(v):
    return 0 if v < 0 else (255 if v > 255 else v)


@lru_cache(maxsize=16384)
def term_color(flags, r, g, b):
    r, g, b = clamp_byte(r), clamp_byte(g), clamp_byte(b)
    is_bg = (flags & FLAG_BG) != 0

    if (flags & FLAG_MODE_256) == 0:
        prefix = "\x1b[48;2;" if is_bg else "\x1b[38;2;"
        return f"{prefix}{r};{g};{b}m"

    ri = _COLOR_STEP_LOOKUP[r]
    gi = _COLOR_STEP_LOOKUP[g]
    bi = _COLOR_STEP_LOOKUP[b]
    rq = COLOR_STEPS[ri]
    gq = COLOR_STEPS[gi]
    bq = COLOR_STEPS[bi]

    gray = int(r * 0.2989 + g * 0.5870 + b * 0.1140 + 0.5)
    if gray > 255:
        gray = 255
    gri = _GRAY_STEP_LOOKUP[gray]
    grq = GRAYSCALE_STEPS[gri]

    color_dist = 0.3 * (rq - r) ** 2 + 0.59 * (gq - g) ** 2 + 0.11 * (bq - b) ** 2
    gray_dist = 0.3 * (grq - r) ** 2 + 0.59 * (grq - g) ** 2 + 0.11 * (grq - b) ** 2

    if color_dist < gray_dist:
        idx = 16 + 36 * ri + 6 * gi + bi
    else:
        idx = 232 + gri

    prefix = "\x1b[48;5;" if is_bg else "\x1b[38;5;"
    return f"{prefix}{idx}m"


def print_image(arr, flags, mask=None):
    h, w = arr.shape[:2]
    out = []
    last_fg = None
    last_bg = None
    needs_reset = True

    fg_flags = flags | FLAG_FG
    bg_flags = flags | FLAG_BG
    noopt = bool(flags & FLAG_NOOPT)

    for y in range(0, h - 7, 8):
        for x in range(0, w - 3, 4):
            if mask is not None and not mask[y:y + 8, x:x + 4].any():
                if not needs_reset:
                    out.append("\x1b[0m")
                    needs_reset = True
                    last_fg = None
                    last_bg = None
                out.append(" ")
                continue

            block = arr[y:y + 8, x:x + 4]
            if noopt:
                cd = create_char_data(block, 0x2584, 0x0000ffff)
            else:
                cd = find_char_data(block, flags)

            bg = cd.bg
            fg = cd.fg
            if needs_reset or bg != last_bg:
                out.append(term_color(bg_flags, bg[0], bg[1], bg[2]))
                last_bg = bg
            if needs_reset or fg != last_fg:
                out.append(term_color(fg_flags, fg[0], fg[1], fg[2]))
                last_fg = fg
            out.append(chr(cd.codepoint))
            needs_reset = False
        out.append("\x1b[0m\n")
        needs_reset = True
        last_fg = None
        last_bg = None

    sys.stdout.write("".join(out))
    sys.stdout.flush()


def load_rgb(path, bg_color):
    img = Image.open(path)
    if img.mode == "P" and "transparency" in img.info:
        img = img.convert("RGBA")
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, bg_color)
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


def fit_size(w, h, max_w, max_h, fill_width=False):
    if fill_width:
        scale = max_w / w
    else:
        scale = min(max_w / w, max_h / h)
    if scale == 1:
        return w, h
    return max(1, int(w * scale)), max(1, int(h * scale))


def resize_image(arr, new_w, new_h):
    img = Image.fromarray(arr).resize((new_w, new_h), Image.Resampling.LANCZOS)
    return np.array(img)


def hex_color(s):
    s = s.lower().removeprefix("0x").removeprefix("#")
    n = int(s, 16)
    return ((n >> 16) & 255, (n >> 8) & 255, n & 255)


def expand_paths(paths):
    result = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            result.extend(str(x) for x in sorted(path.iterdir()) if x.is_file())
        elif path.is_file():
            result.append(str(path))
        else:
            sys.stderr.write(f"Error: Cannot open '{p}'\n")
    return result


def build_parser():
    p = argparse.ArgumentParser(
        description="Terminal Image Viewer (Python port)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-0", dest="noopt", action="store_true",
                   help="no block character adjustment, always use top half block")
    p.add_argument("-2", "--256", dest="mode256", action="store_true",
                   help="use 256-color mode")
    p.add_argument("-c", dest="columns", type=int, default=3,
                   help="thumbnail columns in 'dir' mode (default 3)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("-d", "--dir", dest="mode", action="store_const",
                      const="thumbs", help="force 'dir' mode")
    mode.add_argument("-f", "--full", dest="mode", action="store_const",
                      const="full", help="force 'full' mode")
    p.add_argument("-F", "--fill-width", dest="fill_width", action="store_true",
                   help="scale image to fill the terminal width (may exceed height)")
    p.add_argument("-H", "--height", dest="height", type=int,
                   help="maximum output height in lines")
    p.add_argument("-w", "--width", dest="width", type=int,
                   help="maximum output width in characters")
    p.add_argument("-C", dest="bg_color", type=hex_color,
                   default=(255, 255, 255),
                   help="hex background color for transparent PNG/GIF")
    p.add_argument("images", nargs="+", help="image files or directories")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    flags = 0
    if args.noopt:
        flags |= FLAG_NOOPT
    if args.mode256:
        flags |= FLAG_MODE_256

    file_names = expand_paths(args.images)
    if not file_names:
        return 66

    try:
        term_cols, term_rows = os.get_terminal_size()
    except OSError:
        term_cols, term_rows = 80, 24

    term_rows = max(1, term_rows - 2)

    max_width = (args.width if args.width is not None else term_cols) * 4
    max_height = (args.height if args.height is not None else term_rows) * 8

    mode = args.mode or ("full" if len(file_names) == 1 else "thumbs")
    ret = 0

    if mode == "full":
        for filename in file_names:
            try:
                arr = load_rgb(filename, args.bg_color)
                h, w = arr.shape[:2]
                new_w, new_h = fit_size(w, h, max_width, max_height,
                                        fill_width=args.fill_width)
                if (new_w, new_h) != (w, h):
                    arr = resize_image(arr, new_w, new_h)
                print_image(arr, flags)
            except Exception as e:
                sys.stderr.write(f"Error: '{filename}': {e}\n")
                ret = 65
    else:
        columns = args.columns
        cw = ((max_width // 4) - 2 * (columns - 1)) // columns
        tw = cw * 4
        canvas_w = tw * columns + 2 * 4 * (columns - 1)
        index = 0
        while index < len(file_names):
            canvas = np.zeros((tw, canvas_w, 3), dtype=np.uint8)
            mask = np.zeros((tw, canvas_w), dtype=bool)
            count = 0
            labels = ""
            while index < len(file_names) and count < columns:
                name = file_names[index]
                index += 1
                try:
                    orig = load_rgb(name, args.bg_color)
                    h, w = orig.shape[:2]
                    new_w, new_h = fit_size(w, h, tw, tw)
                    if (new_w, new_h) != (w, h):
                        orig = resize_image(orig, new_w, new_h)
                    x_off = count * (tw + 8) + (tw - new_w) // 2
                    y_off = (tw - new_h) // 2
                    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = orig
                    mask[y_off:y_off + new_h, x_off:x_off + new_w] = True
                    labels += Path(name).name
                    count += 1
                    target_len = count * (cw + 2) - 2
                    labels = labels.ljust(target_len)[:target_len] + "  "
                except Exception:
                    pass
            if count:
                print_image(canvas, flags, mask=mask)
            sys.stdout.write(labels + "\n\n")

    return ret


if __name__ == "__main__":
    sys.exit(main())
