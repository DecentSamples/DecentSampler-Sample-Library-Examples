#!/usr/bin/env python3
"""
dx7_to_dspreset.py  –  Convert a DX7 SysEx bulk dump (.syx) to a
DecentSampler .dspreset file using the fm6op oscillator waveform.

Usage:
    python3 dx7_to_dspreset.py input.syx [output.dspreset]

Supports 32-voice bulk dumps (4104 bytes, the common .syx format).
Multiple concatenated bulk dumps in one file are also handled.

What is translated:
  Algorithm, operator output levels (0–99 → linear amplitude), frequency
  ratios, native DX7 EG rates/levels (R1–R4, L1–L4, range 0–99), and the
  feedback operator + amount.

What is dropped:
  Keyboard level scaling, velocity sensitivity, LFO, pitch EG, and
  transpose.
"""

import sys
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom


# ---------------------------------------------------------------------------
# DX7 output level / EG level (0–99) → linear amplitude.
#
# Derived from Dexed's msfa engine (env.cc scaleoutlevel + fm_core.cc Exp2):
#   scaleoutlevel(n) = 28 + n          for n >= 20
#   scaleoutlevel(n) = LEVELLUT[n]     for n <  20
#   amplitude = 2^((scaleoutlevel(n) - 127) / 8)
# Reference: output_level 99 → scaleoutlevel 127 → amplitude 1.0.
# ---------------------------------------------------------------------------
_LEVELLUT = [0, 5, 9, 13, 17, 20, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 42, 43, 45, 46]

def _scale_level(n: int) -> int:
    return (28 + n) if n >= 20 else _LEVELLUT[n]

def _dx7_amplitude(level: int) -> float:
    """Output level (0-99) → linear amplitude, normalised so level 99 = 1.0."""
    return round(2.0 ** ((_scale_level(level) - 127) / 8.0), 4)

# ---------------------------------------------------------------------------
# For each DX7 algorithm (1–32), which fmOpN (1-indexed) carries feedback.
# Derived from FM6OperatorOscillator.h.
# ---------------------------------------------------------------------------
ALG_FEEDBACK_OP = {
     1: 6,  2: 2,  3: 6,  4: 6,  5: 6,  6: 6,  7: 6,  8: 4,
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
    mode   = b[15] & 0x01           # Bit 0: 0 = ratio, 1 = fixed frequency
    coarse = (b[15] >> 1) & 0x1F   # Bits 1-5: 0-31
    fine   = b[16] & 0x7F          # 0-99
    if mode == 0:
        # Ratio mode: fine is multiplicative — ratio = coarse_mult * (1 + fine/100).
        # Matches Dexed osc_freq(): logfreq += coarsemul[coarse] + 24204406*log(1+fine/100)
        coarse_mult = 0.5 if coarse == 0 else float(coarse)
        ratio     = round(coarse_mult * (1.0 + fine / 100.0), 4)
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
    detune_raw = (b[12] >> 3) & 0x0F  # 0-14, center=7
    detune     = detune_raw - 7         # signed: -7 to +7, 0 = no change
    return {
        "eg_r":         [b[0], b[1], b[2], b[3]],
        "eg_l":         [b[4], b[5], b[6], b[7]],
        "output_level": b[14] & 0x7F,
        "vel_sens":     (b[13] >> 2) & 0x07,  # 0-7, DX7 key velocity sensitivity
        "detune":       detune,                # -7 to +7, 0 = no detune
        "ratio":        ratio,
        "fixed":        fixed,
        "fixed_freq":   fixed_freq,
        "osc_coarse":   coarse,                # raw DX7 coarse integer (0-31)
        "osc_fine":     fine,                  # raw DX7 fine integer (0-99)
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
def _group_attrs(voice: dict, enabled: bool = True, flat_eg: bool = False) -> dict:
    """
    Build the flat attribute dict for a <group> element from a voice dict.
    If flat_eg=True, all operator EGs are replaced with an instant attack /
    hold-at-peak / instant release envelope (R1=99,L1=99 R2=99,L2=99
    R3=0,L3=99 R4=99,L4=0), matching the convention used by dexed_render.py
    and ds_fm_engine.py so the three engines are compared under identical EG
    conditions.
    """
    alg       = voice["algorithm"]
    fb_op     = ALG_FEEDBACK_OP.get(alg, 6)
    # DX7 feedback is exponential (bit-shift): fb_shift = 8 - feedback (for 1-7)
    # DS formula: opFeedback = out * feedback * kFeedbackScale (kFeedbackScale = π²)
    # Correct mapping: DS feedback_param = 2^(dx7_fb - 7)  (range 1/64 to 1.0)
    _fb = voice["feedback"]
    fb_amount = round(0.0 if _fb == 0 else 2.0 ** (_fb - 7), 8)  # DX7 0-7 → DS value

    attrs = {
        "name":    voice["name"],
        "tags":    "fm-preset",
        "enabled": "true" if enabled else "false",
        "volume":  "0.5",
        "attack":  "0.001",
        "decay":   "0",
        "sustain": "1.0",
        "release": "20",
        "fmAlgorithm": str(alg),
    }

    for n, op in voice["ops"].items():
        r     = op["eg_r"]
        l     = op["eg_l"]
        level = _dx7_amplitude(op["output_level"])
        ratio = round(op["ratio"], 4)

        p = f"fmOp{n}"
        if op["fixed"]:
            attrs[f"{p}Mode"]      = "fixed"
            attrs[f"{p}FixedFreq"] = str(op["fixed_freq"])
        else:
            attrs[f"{p}Ratio"] = str(ratio)
        attrs[f"{p}Level"]  = str(level)
        attrs[f"{p}EgType"]  = "dx7"
        if flat_eg:
            # Instant attack to peak, hold forever, instant release —
            # matches dexed_render.py convention for fair FM-only comparison.
            attrs[f"{p}EgRate1"]  = "99"; attrs[f"{p}EgLevel1"] = "99"
            attrs[f"{p}EgRate2"]  = "99"; attrs[f"{p}EgLevel2"] = "99"
            attrs[f"{p}EgRate3"]  = "0";  attrs[f"{p}EgLevel3"] = "99"
            attrs[f"{p}EgRate4"]  = "99"; attrs[f"{p}EgLevel4"] = "0"
        else:
            # Emit native DX7 rate/level EG — no lossy ADSR approximation.
            attrs[f"{p}EgRate1"] = str(r[0])
            attrs[f"{p}EgRate2"] = str(r[1])
            attrs[f"{p}EgRate3"] = str(r[2])
            attrs[f"{p}EgRate4"] = str(r[3])
            attrs[f"{p}EgLevel1"] = str(l[0])
            attrs[f"{p}EgLevel2"] = str(l[1])
            attrs[f"{p}EgLevel3"] = str(l[2])
            attrs[f"{p}EgLevel4"] = str(l[3])
        if n == fb_op:
            attrs[f"{p}Feedback"] = str(fb_amount)
        if op.get("vel_sens", 0) > 0:
            attrs[f"{p}VelocitySensitivity"] = str(op["vel_sens"])
        if op.get("detune", 0) != 0:
            attrs[f"{p}Detune"] = str(op["detune"])

    return attrs


def build_dspreset(voices: list) -> str:
    """Build a .dspreset XML string for the given list of voice dicts."""
    root = ET.Element("DecentSampler", minVersion="1.22.3")

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
        ET.SubElement(opt, "binding",
                      type="general", level="group", tags="fm-preset",
                      parameter="ENABLED",
                      translation="fixed_value",
                      translationValue="false")
        ET.SubElement(opt, "binding",
                      type="general", level="group", position=str(idx),
                      parameter="ENABLED",
                      translation="fixed_value",
                      translationValue="true")

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


def build_single_dspreset(voice: dict, flat_eg: bool = False) -> str:
    """Build a minimal .dspreset XML string for one voice (no dropdown menu).

    flat_eg=True replaces all operator EGs with an instant-attack / hold /
    instant-release envelope for fair FM-engine-only comparison.
    """
    root = ET.Element("DecentSampler", minVersion="1.22.3")

    # ── Groups ──────────────────────────────────────────────────────────
    groups_el = ET.SubElement(root, "groups")
    group = ET.SubElement(groups_el, "group", **_group_attrs(voice, enabled=True, flat_eg=flat_eg))
    ET.SubElement(group, "oscillator",
                  waveform="fm6op",
                  loNote="0", hiNote="127", rootNote="60",
                  loVel="0",  hiVel="127", volume="1.0")

    # ── Pretty-print ────────────────────────────────────────────────────
    raw   = ET.tostring(root, encoding="unicode")
    dom   = minidom.parseString(raw)
    lines = dom.toprettyxml(indent="  ").splitlines()
    return "\n".join(lines[1:])


def _safe_filename(name: str, index: int) -> str:
    """Convert a patch name to a safe filename: '<index+1:02d>_<name>.dspreset'."""
    safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in name).strip()
    safe = safe.replace(" ", "_")
    return f"{index + 1:02d}_{safe}.dspreset"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert a DX7 SysEx bulk dump (.syx) to DecentSampler preset(s).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("syx", help="Input .syx file")
    parser.add_argument("output", nargs="?",
                        help="Output .dspreset file (combined mode, default: <syx>.dspreset)")
    parser.add_argument("--split", action="store_true",
                        help="Write one .dspreset per patch into a directory instead of "
                             "one combined file with a dropdown menu")
    parser.add_argument("--split-dir",
                        help="Directory for split output (default: <syx_stem>_split/)")
    args = parser.parse_args()

    voices = parse_syx(args.syx)
    if not voices:
        print(f"Error: no 32-voice bulk dumps found in {args.syx!r}.")
        print("Only 32-voice bulk-dump format (F0 43 xx 09 …) is supported.")
        sys.exit(1)

    if args.split:
        # ── Split mode: one file per patch ──────────────────────────────
        stem    = os.path.splitext(os.path.basename(args.syx))[0]
        out_dir = args.split_dir or os.path.join(
            os.path.dirname(os.path.abspath(args.syx)), stem + "_split"
        )
        os.makedirs(out_dir, exist_ok=True)
        for idx, voice in enumerate(voices):
            fname    = _safe_filename(voice["name"], idx)
            out_path = os.path.join(out_dir, fname)
            xml_str  = build_single_dspreset(voice)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(xml_str)
            print(f"  [{idx + 1:2d}/{ len(voices)}] {voice['name']:<20s} → {fname}")
        print(f"\nWrote {len(voices)} presets to: {out_dir}")
    else:
        # ── Combined mode: all patches in one file with dropdown ─────────
        out_path = args.output or (os.path.splitext(args.syx)[0] + ".dspreset")
        xml_str  = build_dspreset(voices)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(xml_str)
        print(f"Converted {len(voices)} voice(s)  →  {out_path}")


if __name__ == "__main__":
    main()
