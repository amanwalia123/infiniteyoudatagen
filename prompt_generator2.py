#!/usr/bin/env python3
"""
Generate 4-sentence prompts (man/woman) with scene packs loaded from a directory,
and Flux Dev-friendly positive-only focus cues (no “no blur/bokeh” tokens).

New flags:
  --flux-positive-focus / --flux   : inject positive-only “deep focus” phrases
  --no-negative                    : omit negative prompt in outputs

Also supports:
  --scene-dir <folder with *.json packs>
  --scene-pack <packname> (repeatable)
  --from-attr list_attr_celeba.txt (to sample man/woman)
  --min-size 1024  (adds a size field like "1024x1536")
"""

import argparse, json, random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np

VOWELS = set("aeiou")
SHOTS : List[str] = ["mid-up", "full-body"]
# -------- Built-in scene packs (kept short; extend with --scene-dir) --------
BUILTIN_SCENE_PACKS: Dict[str, Dict[str, List[Tuple[str, float]]]] = {
    "nature_fantasy": {
        "BIOME": [("enchanted forest", 2.0), ("ancient grove", 1.6)],
        "MICRO": [
            ("crystal-clear river over textured, mossy boulders", 1.8),
            ("gnarled roots, ferns, and slick stones", 1.4),
        ],
        "ATMOS": [("soft morning mist", 1.6), ("sunbeams through canopy", 1.4)],
        "TEXTURE": [
            "leaf veins and textured bark are rendered with high micro-contrast",
            "moss, stones, and fern fronds show fine detail across the frame"
        ],
    },
    "urban_cyber": {
        "BIOME": [("neon-lit downtown street", 1.8), ("glass-and-steel city district", 1.4)],
        "MICRO": [
            ("reflections across glass facades and wet pavement", 1.6),
            ("LED billboards, signage, and metal cladding", 1.4),
        ],
        "ATMOS": [("neon haze after rain", 1.5), ("blue-purple ambient glow", 1.3)],
        "TEXTURE": [
            "building edges, clear signage, and window grids are crisply defined",
            "glass seams, metallic panels, and pavement texture remain detailed"
        ],
    },
    "industrial": {
        "BIOME": [("abandoned brick factory", 1.4), ("steel foundry catwalks", 1.3)],
        "MICRO": [
            ("rusted pipes, grates, and riveted beams", 1.6),
            ("oil-slick concrete and mesh flooring", 1.4),
        ],
        "ATMOS": [("dust in sunbeams", 1.2), ("cool industrial fluorescents", 1.1)],
        "TEXTURE": [
            "fine metal seams, rivets, and brick patterns are clearly resolved",
            "distant piping runs and catwalks show consistent detail"
        ],
    }
}

# -------- Flux Dev positive-only focus lexicon (no “no blur” words) --------
FLUX_BASE_POSITIVE = [
    "wide-angle lens",
    "small aperture f/16–f/22",
    "deep focus throughout",
    "everything in frame clearly visible",
    "edge-to-edge clarity",
    "distant background fully detailed",
    "balanced exposure",
    "high micro-contrast",
    "ultra high detail"
]

# Per-scene positive reinforcements (auto-picked if pack name matches)
FLUX_SCENE_HINTS = {
    "nature_fantasy": [
        "distant canopy detail preserved",
        "intricate leaf structure visible",
        "textured bark and moss rendered clearly"
    ],
    "urban_cyber": [
        "clear signage and window grids",
        "distant facades retain crisp geometry",
        "reflections on glass read cleanly"
    ],
    "industrial": [
        "fine rivet lines and seams remain distinct",
        "distant pipes and gantries stay legible",
        "consistent detail across metal textures"
    ]
}

# ------------------ helpers ------------------
# def weighted_choice(items: List[Tuple[str, float]], rng: random.Random) -> str:
#     total = sum(w for _, w in items) or 1.0
#     r = rng.uniform(0, total); acc = 0.0
#     for value, weight in items:
#         acc += weight
#         if r <= acc:
#             return value
#     return items[-1][0]

def weighted_choice(items: List[Tuple[str, float]], rng: random.Random) -> str:
    if not items:
        raise ValueError("No items provided for selection.")
    
    # Unpack values and weights
    values, weights = zip(*items)
    # Compute cumulative weights
    cumulative = np.cumsum(weights)
    total = cumulative[-1]
    
    if total <= 0:
        raise ValueError("Total weight must be positive to make a selection.")
    
    # Generate a random number using the provided RNG
    r = rng.uniform(0, total)
    # Find the index using binary search
    index = np.searchsorted(cumulative, r, side='right')
    return values[index]

def pick_article(phrase: str) -> str:
    w = phrase.strip().lower()
    return "an" if w and w[0] in VOWELS else "a"

def load_scene_packs_file(path: Optional[str]) -> Optional[Dict[str, Dict[str, List[Tuple[str, float]]]]]:
    if not path: return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    packs = data.get("scene_packs", data)
    norm = {}
    for name, pack in packs.items():
        def to_pairs(arr):
            pairs = []
            for x in arr or []:
                if isinstance(x, dict) and "text" in x and "weight" in x:
                    pairs.append((str(x["text"]), float(x["weight"])))
                elif isinstance(x, (list, tuple)) and len(x) == 2:
                    pairs.append((str(x[0]), float(x[1])))
                else:
                    pairs.append((str(x), 1.0))
            return pairs
        norm[name] = {
            "BIOME": to_pairs(pack.get("BIOME", [])),
            "MICRO": to_pairs(pack.get("MICRO", [])),
            "ATMOS": to_pairs(pack.get("ATMOS", [])),
            "TEXTURE": pack.get("TEXTURE", []),
        }
    return norm

def load_scene_dir(path: Optional[str]) -> Optional[Dict[str, Dict[str, List[Tuple[str, float]]]]]:
    if not path: return None
    root = Path(path)
    if not root.exists(): return None
    out: Dict[str, Dict[str, List[Tuple[str, float]]]] = {}
    for f in sorted(root.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        name = data.get("pack_name") or f.stem
        def to_pairs(arr):
            pairs = []
            for x in arr or []:
                if isinstance(x, dict) and "text" in x and "weight" in x:
                    pairs.append((str(x["text"]), float(x["weight"])))
                elif isinstance(x, (list, tuple)) and len(x) == 2:
                    pairs.append((str(x[0]), float(x[1])))
                else:
                    pairs.append((str(x), 1.0))
            return pairs
        out[name] = {
            "BIOME": to_pairs(data.get("BIOME", [])),
            "MICRO": to_pairs(data.get("MICRO", [])),
            "ATMOS": to_pairs(data.get("ATMOS", [])),
            "TEXTURE": data.get("TEXTURE", []),
        }
    return out

def parse_celebA_attr_file(path: str) -> List[str]:
    genders, male_col = [], None
    lines = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    header_idx = None
    for i, ln in enumerate(lines[:3]):
        toks = ln.split()
        if len(toks) > 10 and "Male" in toks:
            header_idx = i; break
    if header_idx is None: header_idx = 1
    headers = lines[header_idx].split()
    try: male_col = headers.index("Male")
    except ValueError: male_col = None
    for ln in lines[header_idx+1:]:
        toks = ln.split()
        if len(toks) < 3: continue
        val = int(toks[male_col+1]) if male_col is not None else 1
        genders.append("man" if val == 1 else "woman")
    return genders

# ------------------ generator ------------------
@dataclass
class GenConfig:
    seed: Optional[int]
    num: int
    force_gender: Optional[str]
    from_attr: Optional[str]
    scene_packs_file: Optional[str]
    scene_dir: Optional[str]
    scene_pack_filters: Optional[List[str]]
    diverse: bool
    out: Optional[str]
    min_size: int
    flux_positive: bool
    no_negative: bool

class PromptGenerator:
    def __init__(self, cfg: GenConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        # assemble scene packs
        self.packs = {k: v.copy() for k, v in BUILTIN_SCENE_PACKS.items()}
        dirpacks = load_scene_dir(cfg.scene_dir)
        if dirpacks:
            for k, v in dirpacks.items():
                cur = self.packs.get(k, {"BIOME": [], "MICRO": [], "ATMOS": [], "TEXTURE": []})
                cur["BIOME"].extend(v.get("BIOME", []))
                cur["MICRO"].extend(v.get("MICRO", []))
                cur["ATMOS"].extend(v.get("ATMOS", []))
                cur["TEXTURE"] = cur.get("TEXTURE", []) + v.get("TEXTURE", [])
                self.packs[k] = cur
        filepacks = load_scene_packs_file(cfg.scene_packs_file)
        if filepacks:
            for k, v in filepacks.items():
                cur = self.packs.get(k, {"BIOME": [], "MICRO": [], "ATMOS": [], "TEXTURE": []})
                cur["BIOME"].extend(v.get("BIOME", []))
                cur["MICRO"].extend(v.get("MICRO", []))
                cur["ATMOS"].extend(v.get("ATMOS", []))
                cur["TEXTURE"] = cur.get("TEXTURE", []) + v.get("TEXTURE", [])
                self.packs[k] = cur

        self.used_tuples = set()
        self.gender_pool = parse_celebA_attr_file(cfg.from_attr) if cfg.from_attr else None

    def _pick_gender(self) -> str:
        if self.cfg.force_gender in ("man", "woman"): return self.cfg.force_gender
        if self.gender_pool: return self.rng.choice(self.gender_pool)
        return self.rng.choice(["man", "woman"])

    def _build_scene_sources(self):
        chosen = self.cfg.scene_pack_filters or list(self.packs.keys())
        biome, micro, atmos, textures, packnames = [], [], [], [], []
        for pack in chosen:
            data = self.packs.get(pack, {})
            if data: packnames.append(pack)
            biome.extend(data.get("BIOME", []))
            micro.extend(data.get("MICRO", []))
            atmos.extend(data.get("ATMOS", []))
            textures.extend(data.get("TEXTURE", []))
        if not biome: biome = [("textured environment", 1.0)]
        if not micro: micro = [("clearly defined background objects", 1.0)]
        if not atmos: atmos = [("soft natural light", 1.0)]
        if not textures: textures = ["background textures are rendered with clarity"]
        return biome, micro, atmos, textures, packnames

    def _compose_scene(self):
        biome_pool, micro_pool, atmos_pool, textures, packnames = self._build_scene_sources()
        biome = weighted_choice(biome_pool, self.rng)
        micro = weighted_choice(micro_pool, self.rng)
        atmos = weighted_choice(atmos_pool, self.rng)
        tex = self.rng.choice(textures)
        # heuristic main pack used for scene hints
        pack_for_hints = next((p for p in packnames if p in ("nature_fantasy","urban_cyber","industrial")), packnames[0] if packnames else "")
        return biome, micro, atmos, tex, pack_for_hints

    def _unique_ok(self, shot: str, biome: str, micro: str, atmos: str) -> bool:
        key = (shot, biome, micro, atmos)
        if not self.cfg.diverse: return True
        if key in self.used_tuples: return False
        self.used_tuples.add(key); return True

    # ----- Sentences (Flux-positive wording if enabled) -----
    # def _first_sentence(self, subject: str, biome: str) -> str:
    #     art = pick_article(biome)
    #     return f"First, mid-up photography of {subject} in {art} {biome}."
    
    def _first_sentence(self, subject: str, biome: str, shot: str) -> str:
        art = pick_article(biome)
        return f"First, {shot} photography of {subject} in {art} {biome}"
    

    def _second_sentence(self, flux: bool, pack_hint: str) -> str:
        if not flux:
            return "Second, vertical orientation, direct perspective, (wide angle view:1.2)."
        # positive-only focus phrases
        add = self.rng.sample(FLUX_BASE_POSITIVE, k=4)
        scene_add = self.rng.sample(FLUX_SCENE_HINTS.get(pack_hint, []), k=min(1, len(FLUX_SCENE_HINTS.get(pack_hint, []))))
        core = ["vertical orientation", "direct perspective", "wide-angle view 1.2"]
        return "Second, " + ", ".join(core + add + scene_add) + "."

    def _third_sentence(self, micro: str, atmos: str, flux: bool) -> str:
        art = pick_article(micro)
        if not flux:
            return f"Third, {art} {micro} in the background, with {atmos}, all perfectly in focus."
        # positive-only, clarity on distant objects
        return f"Third, {art} {micro} in the background, with {atmos}, distant objects clearly rendered, edge-to-edge clarity."

    def _fourth_sentence(self, texture_line: str, flux: bool) -> str:
        if not flux:
            return f"Fourth, {texture_line}."
        return f"Fourth, {texture_line}, consistent detail across near and far surfaces, cinematic realism, ultra high detail."

    def _size_str(self) -> str:
        w = max(1024, int(self.cfg.min_size))
        h = int(round(w * 1.5))  # portrait bias to encourage detail
        return f"{w}x{h}"

    def negative_prompt(self) -> str:
        if self.cfg.no_negative: return ""
        # keep safe/minimal; no blur terms here to avoid priming
        return "lowres, text, watermark, logo, artifacts, oversaturated, deformed anatomy, distorted proportions"

    def generate_one(self) -> dict:
        subject = self._pick_gender()
        shot = random.choice(SHOTS)
        for _ in range(24):
            biome, micro, atmos, tex, hint = self._compose_scene()
            if self._unique_ok(shot, biome, micro, atmos):
                break
        s1 = self._first_sentence(subject, biome, shot)
        s2 = self._second_sentence(self.cfg.flux_positive, hint)
        s3 = self._third_sentence(micro, atmos, self.cfg.flux_positive)
        s4 = self._fourth_sentence(tex, self.cfg.flux_positive)
        item = {
            "prompt": f"{s1} {s2} {s3} {s4}",
            "size": self._size_str()
        }
        neg = self.negative_prompt()
        if neg: item["negative_prompt"] = neg
        return item

    def generate_many(self, n: int) -> List[dict]:
        return [self.generate_one() for _ in range(n)]

# ------------------ CLI ------------------
def build_argparser():
    ap = argparse.ArgumentParser(description="Flux Dev-friendly prompt generator (man/woman) with scene directory packs.")
    ap.add_argument("-n", "--num", type=int, default=5)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--force-gender", type=str, choices=["man", "woman"])
    ap.add_argument("--from-attr", type=str, default=None, help="Path to CelebA attributes file (list_attr_celeba.txt).")
    ap.add_argument("--scene-dir", type=str, default=None, help="Directory of *.json scene packs (one per file).")
    ap.add_argument("--scene-packs-file", type=str, default=None, help="(Legacy) combined JSON with multiple packs.")
    ap.add_argument("--scene-pack", action="append", default=None, help="Restrict to these pack names (repeatable).")
    ap.add_argument("--diverse", action="store_true", help="Avoid repeating (shot, biome, micro, atmos) combos.")
    ap.add_argument("--min-size", type=int, default=1024, help="Minimum dimension; outputs size like 1024x1536.")
    ap.add_argument("--flux-positive-focus", "--flux", action="store_true", help="Use positive-only deep-focus phrasing for Flux Dev.")
    ap.add_argument("--no-negative", action="store_true", help="Omit negative prompt.")
    ap.add_argument("-o", "--out", type=str, default=None, help="Write to .txt or .json")
    return ap

def main():
    args = build_argparser().parse_args()
    cfg = GenConfig(
        seed=args.seed,
        num=max(1, int(args.num)),
        force_gender=args.force_gender,
        from_attr=args.from_attr,
        scene_packs_file=args.scene_packs_file,
        scene_dir=args.scene_dir,
        scene_pack_filters=args.scene_pack,
        diverse=args.diverse,
        out=args.out,
        min_size=args.min_size,
        flux_positive=args.flux_positive_focus,
        no_negative=args.no_negative,
    )
    gen = PromptGenerator(cfg)
    items = gen.generate_many(cfg.num)

    if args.out:
        out = args.out
        if out.lower().endswith(".json"):
            Path(out).write_text(json.dumps({"items": items}, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            # write prompts lines; sidecar a .size and (optional) .neg file
            Path(out).write_text("\n".join(i["prompt"] for i in items) + "\n", encoding="utf-8")
            Path(out + ".size.txt").write_text("\n".join(i["size"] for i in items) + "\n", encoding="utf-8")
            if not args.no_negative:
                Path(out + ".neg.txt").write_text(gen.negative_prompt() + "\n", encoding="utf-8")
        print(f"Wrote {len(items)} to {out}")
    else:
        for i in items:
            print(i["prompt"])
            print("SIZE:", i["size"])
            if "negative_prompt" in i:
                print("\n NEGATIVE:", i["negative_prompt"])

if __name__ == "__main__":
    main()
