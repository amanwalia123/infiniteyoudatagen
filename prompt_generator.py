
#!/usr/bin/env python3
"""
prompt_simple_gender_generator.py

Generates 4-sentence prompts in your original style, with:
- Subject limited to "man" or "woman"
- Optional gender sampling from CelebA/CelebA-HQ attributes file (list_attr_celeba.txt)
- Scene coherence (Part 1 ↔ Part 3): BIOME ↔ MICRO objects in the background
- Hard no-blur language: deep depth of field (f/8–f/11), perfect focus
- Scene packs: built-in + external JSON file
"""
import argparse, json, random, re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np 

VOWELS = set("aeiou")
SHOTS : List[str] = ["mid-up", "full-body"]
BUILTIN_SCENE_PACKS: Dict[str, Dict[str, List[Tuple[str, float]]]] = {
    "nature_fantasy": {
        "BIOME": [("enchanted forest", 1.0), ("ancient grove", 1.6), ("bioluminescent woodland", 1.3)],
        "MICRO": [
            ("crystal-clear river over textured, mossy boulders", 1.8),
            ("gnarled roots, ferns, and slick stones", 1.4),
            ("small stone footbridge over a brook", 1.3)
        ],
        "ATMOS": [("soft morning mist", 1.6), ("sunbeams through canopy", 1.4), ("drifting fireflies", 1.2)],
        "TEXTURE": [
            "tree leaves and bark patterns are crystal-sharp",
            "fern fronds and moss textures are razor-detailed",
            "stone and water ripples are richly textured and crisp"
        ],
    },
    "urban_cyber": {
        "BIOME": [("neon-lit downtown street", 1.8), ("glass-and-steel city district", 1.4)],
        "MICRO": [
            ("reflections across glass facades and wet pavement", 1.6),
            ("LED billboards, signage, and metal cladding", 1.4),
            ("aerial walkways and window grids", 1.2),
        ],
        "ATMOS": [("neon haze after rain", 1.5), ("blue-purple ambient glow", 1.3)],
        "TEXTURE": [
            "building edges, signage, and window frames are crystal-clear",
            "metal panels, rivets, and pavement texture are perfectly sharp",
            "glass seams and architectural lines are razor-defined"
        ],
    },
    "coastal": {
        "BIOME": [("rocky coastline", 1.6), ("windswept beach dunes", 1.4)],
        "MICRO": [
            ("tidal pools among barnacled stones", 1.6),
            ("weathered driftwood and sea grass", 1.4),
        ],
        "ATMOS": [("golden hour sea spray", 1.4), ("overcast, moody surf", 1.2)],
        "TEXTURE": [
            "pebbles, shells, and kelp strands are crisply defined",
            "foam lace and wet rock textures are rendered in perfect focus"
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
            "rust flakes, metal seams, and grating are sharply resolved",
            "brick patterns and pipe fittings appear crystal-sharp"
        ],
    },
    "european_oldtown": {
        "BIOME": [("cobblestoned old town", 1.5), ("medieval market square", 1.3)],
        "MICRO": [
            ("ivy-clad stone facades and timber frames", 1.5),
            ("arched passageways and lanterns", 1.3),
        ],
        "ATMOS": [("cloudy soft light", 1.2), ("warm window glows at dusk", 1.1)],
        "TEXTURE": [
            "cobblestones, brickwork, and wrought iron details are crisp",
            "wood grain and stone textures are rendered with fine detail"
        ],
    },
    "rainforest": {
        "BIOME": [("tropical rainforest", 1.6), ("cloud forest", 1.4)],
        "MICRO": [
            ("buttress roots, vines, and mossy logs", 1.6),
            ("bromeliads and lianas over slick stones", 1.4),
        ],
        "ATMOS": [("humid mist with diffuse light", 1.5), ("sunshafts after rainfall", 1.3)],
        "TEXTURE": [
            "leaf veins, moss, and bark textures are exquisitely sharp",
            "water droplets and plant fibers are finely detailed"
        ],
    },
    "desert": {
        "BIOME": [("sandstone canyon", 1.4), ("dune field", 1.4)],
        "MICRO": [
            ("rippling dune crests and varnished rocks", 1.5),
            ("cracked earth and wind-carved ridges", 1.3),
        ],
        "ATMOS": [("clear, hard sunlight", 1.3), ("long blue shadows at dusk", 1.2)],
        "TEXTURE": [
            "sand grains, rock striations, and crust patterns are razor-sharp",
            "wind textures and stone edges are perfectly resolved"
        ],
    }
}
# def weighted_choice(items: List[Tuple[str, float]], rng: random.Random) -> str:
#     total = sum(w for _, w in items) or 1.0
#     r = rng.uniform(0, total)
#     acc = 0.0
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
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
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
def parse_celebA_attr_file(path: str) -> List[str]:
    genders = []
    male_col = None
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    header_idx = None
    for i, ln in enumerate(lines[:3]):
        toks = ln.split()
        if len(toks) > 10 and "Male" in toks:
            header_idx = i
            break
    if header_idx is None:
        header_idx = 1
    headers = lines[header_idx].split()
    try:
        male_col = headers.index("Male")
    except ValueError:
        male_col = None
    for ln in lines[header_idx+1:]:
        toks = ln.split()
        if len(toks) < 3:
            continue
        val = int(toks[male_col+1]) if male_col is not None else 1
        genders.append("man" if val == 1 else "woman")
    return genders
@dataclass
class GenConfig:
    seed: Optional[int]
    num: int
    force_gender: Optional[str]
    from_attr: Optional[str]
    scene_packs_file: Optional[str]
    scene_pack_filters: Optional[List[str]]
    diverse: bool
    out: Optional[str]
    image_size: Tuple[int, int] = (1536, 1024)  # Default image size for prompts

class PromptGenerator:
    def __init__(self, cfg: GenConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.packs = BUILTIN_SCENE_PACKS.copy()
        extra = load_scene_packs_file(cfg.scene_packs_file)
        if extra:
            for k, v in extra.items():
                cur = self.packs.get(k, {"BIOME": [], "MICRO": [], "ATMOS": [], "TEXTURE": []})
                cur["BIOME"].extend(v.get("BIOME", []))
                cur["MICRO"].extend(v.get("MICRO", []))
                cur["ATMOS"].extend(v.get("ATMOS", []))
                cur_text = cur.get("TEXTURE", [])
                ext_text = v.get("TEXTURE", [])
                if ext_text:
                    cur_text.extend(ext_text)
                cur["TEXTURE"] = cur_text
                self.packs[k] = cur
        self.used_tuples = set()
        self.gender_pool = None
        if cfg.from_attr:
            try:
                self.gender_pool = parse_celebA_attr_file(cfg.from_attr)
            except Exception:
                self.gender_pool = None
    
    def _pick_gender(self) -> str:
        if self.cfg.force_gender in ("man", "woman"):
            return self.cfg.force_gender
        if self.gender_pool:
            return self.rng.choice(self.gender_pool)
        # return self.rng.choice(["man", "woman"])
        return "person"  # fallback
    
    def _build_scene_sources(self):
        chosen = self.cfg.scene_pack_filters or list(self.packs.keys())
        biome, micro, atmos, textures = [], [], [], []
        for pack in chosen:
            data = self.packs.get(pack, {})
            biome.extend(data.get("BIOME", []))
            micro.extend(data.get("MICRO", []))
            atmos.extend(data.get("ATMOS", []))
            textures.extend(data.get("TEXTURE", []))
        if not biome: biome = [("textured environment", 1.0)]
        if not micro: micro = [("clearly defined background objects", 1.0)]
        if not atmos: atmos = [("soft natural light", 1.0)]
        if not textures: textures = ["background textures are crystal-sharp"]
        return biome, micro, atmos, textures
    def _compose_scene(self):
        biome_pool, micro_pool, atmos_pool, textures = self._build_scene_sources()
        biome = weighted_choice(biome_pool, self.rng)
        micro = weighted_choice(micro_pool, self.rng)
        atmos = weighted_choice(atmos_pool, self.rng)
        tex = self.rng.choice(textures)
        return biome, micro, atmos, tex
    
    def _unique_ok(self, shot: str, biome: str, micro: str, atmos: str) -> bool:
        key = (shot, biome, micro, atmos)
        if not self.cfg.diverse:
            return True
        if key in self.used_tuples:
            return False
        self.used_tuples.add(key)
        return True
    
    def _first_sentence(self, subject: str, biome: str, shot: str) -> str:
        art = pick_article(biome)
        return f"First, {shot} photography of {subject} in {art} {biome} ."
    
    # def _second_sentence(self) -> str:
    #     return "Second, vertical orientation, direct perspective, front-facing (wide angle view:1.2), no tilt or distortion."

    def _second_sentence(self) -> str:
        # stronger “no blur” phrasing + small aperture
        return ("Second, vertical orientation, direct perspective, front-facing  (wide angle view:1.2), "
                "small aperture (f/16–f/22).")

    # def _third_sentence(self, micro: str, atmos: str) -> str:
    #     art = pick_article(micro)
    #     return f"Third, {art} {micro} in the background, with {atmos}, all perfectly in focus."
    
    def _third_sentence(self, micro: str, atmos: str) -> str:
        art = pick_article(micro)
        return (f"Third, {art} {micro} in the background, with {atmos}, "
                "background fully in focus, uniformly sharp edge-to-edge.")

    # def _fourth_sentence(self, texture_line: str) -> str:
    #     return f"Fourth, {texture_line} are very sharp and in focus, with no blur or bokeh anywhere."

    def _fourth_sentence(self, texture_line: str) -> str:
        return (f"Fourth, {texture_line}, high micro-contrast and crisp detail;")
    
    # Add this helper:
    def negative_prompt(self) -> str:
        return ("no bokeh, no blur, no defocus, no shallow depth of field, "
                "no soft focus, no gaussian blur, no motion blur, no background blur, no tilt-shift")

    def generate_one(self) -> str:
        subject = self._pick_gender()
        shot = random.choice(SHOTS)
        for _ in range(24):
            biome, micro, atmos, tex = self._compose_scene()
            if self._unique_ok(shot, biome, micro, atmos):
                break
        s1 = self._first_sentence(subject, biome, shot)
        s2 = self._second_sentence()
        s3 = self._third_sentence(micro, atmos)
        s4 = self._fourth_sentence(tex)
        return f"{s1} {s2} {s3} {s4}."
    
    def generate_many(self, n: int) -> List[str]:
        return [self.generate_one() for _ in range(n)]

def build_argparser():
    ap = argparse.ArgumentParser(description="Generate original-style prompts with subject = man/woman and coherent scenes.")
    ap.add_argument("-n", "--num", type=int, default=5)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--force-gender", type=str, choices=["man", "woman"])
    ap.add_argument("--from-attr", type=str, default=None, help="Path to CelebA attributes file (list_attr_celeba.txt).")
    ap.add_argument("--scene-packs-file", type=str, default=None, help="JSON file with scene packs.")
    ap.add_argument("--scene-pack", action="append", default=None, help="Restrict to these packs (repeatable).")
    ap.add_argument("--diverse", action="store_true", help="Avoid repeating (shot, biome, micro, atmos) combos.")
    ap.add_argument("-o", "--out", type=str, default=None, help="Write prompts to .txt or .json")
    return ap
def main():
    args = build_argparser().parse_args()
    cfg = GenConfig(
        seed=args.seed,
        num=max(1, int(args.num)),
        force_gender=args.force_gender,
        from_attr=args.from_attr,
        scene_packs_file=args.scene_packs_file,
        scene_pack_filters=args.scene_pack,
        diverse=args.diverse,
        out=args.out,
    )
    gen = PromptGenerator(cfg)
    prompts = gen.generate_many(cfg.num)
    if args.out:
        out = args.out
        if out.lower().endswith(".json"):
            with open(out, "w", encoding="utf-8") as f:
                json.dump({"prompts": prompts}, f, ensure_ascii=False, indent=2)
        else:
            with open(out, "w", encoding="utf-8") as f:
                for p in prompts:
                    f.write(p + "\n")
        print(f"Wrote {len(prompts)} prompts to {out}")
    else:
        for p in prompts:
            print(p)
if __name__ == "__main__":
    main()
