#!/usr/bin/env python3
"""
dx7_to_dspreset.py  –  Convert a DX7 SysEx bulk dump (.syx) to a
DecentSampler .dspreset file using the fm6op oscillator waveform.

Usage:
    python3 dx7_to_dspreset.py input.syx [output.dspreset]

Supports 32-voice bulk dumps (4104 bytes, the common .syx format).
Multiple concatenated bulk dumps in one file are also handled.

What is translated:
  Algorithm, operator output levels, frequency ratios, EG rates/levels
  (approximated as ADSR), and the feedback operator + amount.

What is dropped:
  Keyboard level scaling, velocity sensitivity, LFO, pitch EG, transpose,
  and fixed-frequency operators (those fall back to ratio 1×).
"""

import sys
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom


# ---------------------------------------------------------------------------
# DX7 EG rate (0–99) → approximate time in seconds.
# Maps exponentially: rate 99 ≈ 1 ms, rate 0 ≈ 20 s.
# ---------------------------------------------------------------------------
def eg_rate_to_sec(rate: int) -> float:
    rate = max(0, min(99, int(rate)))
    if rate >= 99:
        return 0.001
    return round(0.001 * (20_000 ** ((99 - rate) / 99.0)), 4)


# ---------------------------------------------------------------------------
# For each DX7 algorithm (1–32), which fmOpN (1-indexed) carries feedback.
# Derived from FM6OperatorOscillator.h.
# ---------------------------------------------------------------------------
ALG_FEEDBACK_OP = {
     1: 6,  2: 2,  3: 6,  4: 4,  5: 6,  6: 5,  7: 6,  8: 4,
     9: 2, 10: 3, 11: 6, 12: 2, 13: 6, 14: 6, 15: 2, 16: 6,
    17: 2, 18: 3, 19: 6, 20: 3, 21: 3, 22: 6, 23: 6, 24: 6,
    25: 6, 26: 6, 27: 3, 28: 5, 29: 6, 30: 5, 31: 6, 32: 6,
}


# ---------------------------------------------------------------------------
# Packed voice parsing
# ---------------------------------------------------------------------------
def _unpack_operator(voice_bytes: bytes, dx7_op: int) -> dict:
    """
    Unpack one operator from a 128-byte packed DX7 voice.
    dx7_op is 1-indexed (OP1 = carrier end, OP6 = top modulator).
    The packed format stores operators in reverse: OP6 first, OP1 last.
    """
    base = (6 - dx7_op) * 17
    b = voice_bytes[base:base + 17]
    mode   = b[15] & 0x01          # 0 = ratio, 1 = fixed frequency
    coarse = (b[15] >> 1) & 0x1F  # 0-31
    fine   = b[16] & 0x7F          # 0-99
    if mode == 0:
        # Ratio mode: coarse 0 = 0.5×, else coarse + fine/100
        ratio     = 0.5 if coarse == 0 else float(coarse) + fine / 100.0
        fixed     = False
        fixed_freq = None
    else:
        # Fixed-frequency mode.  The DX7 encodes frequency using only the
        # lower 2 bits of coarse as a decade selector (0→1Hz, 1→10Hz,
        # 2→100Hz, 3→1000Hz), with fine providing fractional scaling:
        #   freq_Hz = 10^(coarse & 3) * 10^(fine / 100)
        # This matches Dexed's OperatorEditor display formula:
        #   freq = pow(10, coarse & 3) * exp(ln(10) * fine / 100)
        ratio      = 1.0   # ignored when fixed=True
        fixed      = True
        fixed_freq = round((10.0 ** (coarse & 3)) * (10.0 ** (fine / 100.0)), 4)
    return {
        "eg_r":         [b[0], b[1], b[2], b[3]],
        "eg_l":         [b[4], b[5], b[6], b[7]],
        "output_level": b[14] & 0x7F,
        "ratio":        ratio,
        "fixed":        fixed,
        "fixed_freq":   fixed_freq,
    }


def _unpack_voice(voice_bytes: bytes) -> dict:
    """Return a dict describing one 128-byte packed DX7 voice."""
    g         = voice_bytes[102:]
    algorithm = (g[8] & 0x1F) + 1   # 1–32
    feedback  =  g[9] & 0x07         # 0–7
    raw_name  = voice_bytes[118:128]  # bytes 16–25 of global section
    name      = "".join(chr(b & 0x7F) for b in raw_name).strip()
    return {
        "name":      name or "Untitled",
        "algorithm": algorithm,
        "feedback":  feedback,
        "ops":       {n: _unpack_operator(voice_bytes, n) for n in range(1, 7)},
    }


def parse_syx(path: str) -> list:
    """
    Return a list of voice dicts from a .syx file.
    Scans for all F0 43 xx 09 bulk-dump headers in the file.
    """
    data = open(path, "rb").read()
    voices = []
    offset = 0
    while offset <= len(data) - 4104:
        if (data[offset] == 0xF0
                and data[offset + 1] == 0x43
                and data[offset + 3] == 0x09):
            for i in range(32):
                start = offset + 6 + i * 128
                voices.append(_unpack_voice(data[start:start + 128]))
            offset += 4104
        else:
            offset += 1
    return voices


# ---------------------------------------------------------------------------
# Preset generation
# ---------------------------------------------------------------------------
def _group_attrs(voice: dict, enabled: bool) -> dict:
    """
    Build the flat attribute dict for a <group> element from a voice dict.
    """
    alg       = voice["algorithm"]
    fb_op     = ALG_FEEDBACK_OP.get(alg, 6)
    fb_amount = round(voice["feedback"] / 7.0 * 0.5, 4)  # 0–7  →  0.0–0.5

    attrs = {
        "enabled": "true" if enabled else "false",
        "volume":  "0.5",
        "attack":  "0.001",
        "decay":   "0",
        "sustain": "1.0",
        "release": "20",
        "fmAlgorithm": str(alg),
    }

    for n, op in voice["ops"].items():
        r       = op["eg_r"]
        l       = op["eg_l"]
        level   = round(op["output_level"] / 99.0, 4)
        ratio   = round(op["ratio"], 4)
        attack  = eg_rate_to_sec(r[0])
        decay   = eg_rate_to_sec(r[1])
        sustain = round(l[2] / 99.0, 4)  # EG L3 = sustained target level
        release = eg_rate_to_sec(r[3])

        p = f"fmOp{n}"
        if op["fixed"]:
            attrs[f"{p}Mode"]      = "fixed"
            attrs[f"{p}FixedFreq"] = str(op["fixed_freq"])
        else:
            attrs[f"{p}Ratio"] = str(ratio)
        attrs[f"{p}Level"]   = str(level)
        attrs[f"{p}Attack"]  = str(attack)
        attrs[f"{p}Decay"]   = str(decay)
        attrs[f"{p}Sustain"] = str(sustain)
        attrs[f"{p}Release"] = str(release)
        if n == fb_op:
            attrs[f"{p}Feedback"] = str(fb_amount)

    return attrs


def build_dspreset(voices: list) -> str:
    """Build a .dspreset XML string for the given list of voice dicts."""
    root = ET.Element("DecentSampler", minVersion="1.0.0")

    # ── UI ──────────────────────────────────────────────────────────────
    ui  = ET.SubElement(root, "ui", width="812", height="375",
                        bgColor="#FF1A1A2E")
    tab = ET.SubElement(ui, "tab", name="main")
    ET.SubElement(tab, "label",
                  x="16", y="8", width="200", height="26",
                  text="Patch", textColor="#FFFFFFFF", fontSize="16")
    menu = ET.SubElement(tab, "menu",
                         x="16", y="38", width="500", height="30",
                         value="1", style="classic")

    for idx, voice in enumerate(voices):
        opt = ET.SubElement(menu, "option", name=voice["name"])
        for j in range(len(voices)):
            ET.SubElement(opt, "binding",
                          type="general", level="group", position=str(j),
                          parameter="ENABLED",
                          translation="fixed_value",
                          translationValue="true" if j == idx else "false")

    # ── Groups ──────────────────────────────────────────────────────────
    groups_el = ET.SubElement(root, "groups")
    for idx, voice in enumerate(voices):
        group = ET.SubElement(groups_el, "group",
                              **_group_attrs(voice, enabled=(idx == 0)))
        ET.SubElement(group, "oscillator",
                      waveform="fm6op",
                      loNote="0", hiNote="127", rootNote="60",
                      loVel="0",  hiVel="127", volume="1.0")

    # ── Pretty-print (no XML declaration) ───────────────────────────────
    raw  = ET.tostring(root, encoding="unicode")
    dom  = minidom.parseString(raw)
    lines = dom.toprettyxml(indent="  ").splitlines()
    return "\n".join(lines[1:])  # strip the <?xml …?> declaration line


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    syx_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) >= 3 else (
        os.path.splitext(syx_path)[0] + ".dspreset"
    )

    voices = parse_syx(syx_path)
    if not voices:
        print(f"Error: no 32-voice bulk dumps found in {syx_path!r}.")
        print("Only 32-voice bulk-dump format (F0 43 xx 09 …) is supported.")
        sys.exit(1)

    xml_str = build_dspreset(voices)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml_str)

    print(f"Converted {len(voices)} voice(s)  →  {out_path}")


if __name__ == "__main__":
    main()
