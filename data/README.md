# hCaptcha Image-Grid Dataset

Labeled image dataset for the vision solver (`vision_solver.py`) — the model
that reads the hCaptcha prompt ("select all images with a **bus**", "…a
**car**", "…**traffic lights**"…) and picks the matching grid tiles.

All images are procedurally rendered (street/sky scenes, randomized pose,
colour, lighting and camera noise), fully deterministic per seed, no network
and no licence issues. Each class lives in its own folder (ImageFolder style).

## Layout

```
data/
  bus/bus_00000.jpg … bus_00599.jpg
  car/car_00000.jpg …
  red_light/red_light_00000.jpg …
  …
  manifest.jsonl     # one line per image: {"image","label","class_id","prompt"}
  _preview.jpg       # contact sheet of every class
```

## Classes (13)

| folder           | label         | hCaptcha prompt it covers          |
|------------------|---------------|------------------------------------|
| `bus/`           | bus           | "buses"                            |
| `car/`           | car           | "cars"                             |
| `truck/`         | truck         | "trucks"                           |
| `train/`         | train         | "trains"                           |
| `bicycle/`       | bicycle       | "bicycles"                         |
| `motorcycle/`    | motorcycle    | "motorcycles"                      |
| `boat/`          | boat          | "boats"                            |
| `airplane/`      | airplane      | "airplanes"                        |
| `traffic_light/` | traffic light | "traffic lights" (any signal)      |
| `red_light/`     | red light     | "red lights" (red lamp lit)        |
| `crosswalk/`     | crosswalk     | "crosswalks"                       |
| `fire_hydrant/`  | fire hydrant  | "fire hydrants"                    |
| `parking_meter/` | parking meter | "parking meters"                   |

`red_light` is kept separate from `traffic_light` on purpose: in `red_light`
the red lamp is **always** lit, while `traffic_light` only ever shows a dim
3-lamp signal or a yellow/green lamp — the two labels never overlap.

## Regenerate / scale up

```bash
# default: 600 per class (7,800 images, this checkout)
python make_dataset.py

# tens of thousands (13 × 3000 = 39,000 images):
python make_dataset.py --per_class 3000

# bigger tiles (fine-tune on 128×128 instead of 96×96):
python make_dataset.py --per_class 3000 --size 128

# a single class:
python make_dataset.py --classes bus,car,red_light --per_class 2000

# different output folder:
python make_dataset.py --out my_dataset --per_class 1000
```

Everything is seeded (`--seed 1`), so the same command reproduces the exact
same images anywhere.

## Use it

The folder-per-class layout works directly with a standard image classifier
(PyTorch `ImageFolder` / `datasets.load_dataset("imagefolder")`). For the
repo's SmolVLM fine-tuner style (`train/fine_tune_smolvlm.py`), convert the
manifest into prompt/answer pairs:

```python
import json
with open("data/manifest.jsonl") as f:
    for line in f:
        r = json.loads(line)
        # prompt -> "Does this image contain a {label}?"
        # answer -> "yes" / "no"
```

For each image you can also mint the negative counterpart by pairing it with
a different class's prompt ("Does this contain a truck?" → "no"), which is
exactly the tile-selection signal the solver needs to generalize.
