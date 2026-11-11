# ImageSpace settings

The [README](../README.md) covers everyday use. This page lists the settings you can change. Each one is added to the `imagespace` command, for example:

```bash
imagespace -i my-images -o ./output --seed 42
```

You can also see this list by running `imagespace --help`.

## Input and output

| Setting | Default | What it does |
|---------|---------|--------------|
| `-i`, `--input` | required | Folder of images |
| `-o`, `--output` | required | Folder where the website is written |
| `--metadata` | none | CSV file describing the images (needs a `filename` column) |
| `--label-column` | none | Metadata column used to describe each cluster, such as `style` |
| `--label-mixed-threshold` | 0.5 | A cluster is marked "mixed" when its most common value covers less than this share of it |

## Model, map, and clusters

| Setting | Default | What it does |
|---------|---------|--------------|
| `--clip-model` | `openai/clip-vit-base-patch32` | The Hugging Face CLIP model that compares the images |
| `--umap-neighbors` | 15 | UMAP neighbors for the map |
| `--umap-min-dist` | 0.1 | UMAP minimum distance for the map |
| `--cluster-neighbors` | 30 | UMAP neighbors for the space used to find clusters |
| `--cluster-dims` | 10 | Dimensions of that space |
| `--min-cluster-size` | about 1% of the collection | Smallest group HDBSCAN will call a cluster |
| `--hdbscan-min-samples` | 5 | How dense a region must be to count as a cluster |
| `--hdbscan-selection-method` | `leaf` | How HDBSCAN picks clusters: `leaf` or `eom` |
| `--seed` | none | A number that makes the map and clusters come out the same every run (slower) |

Images that fit no cluster are never forced into one. They stay unassigned, and the viewer's Noise control shows them.

## Image size and quality

| Setting | Default | What it does |
|---------|---------|--------------|
| `--thumb-size` | 128 | Thumbnail size in pixels |
| `--atlas-size` | 4096 | Size of the image sheets the viewer loads |
| `--quality` | 60 | Thumbnail quality (1–100) |
| `--no-hd` | off | Skip the low-resolution previews that make the site load faster |
| `--preview-quality` | 40 | Quality of those previews |

## Faster reruns

| Setting | What it does |
|---------|--------------|
| `--cache-dir` | Saves the model's reading of the images in a folder, so later runs skip the slowest step |
| `--relayout` | Rebuilds only the map and clusters, using the saved reading |
| `--data-only` | Writes only the `data/` folder, without the viewer |

```bash
# First run: save the model's reading of the images
imagespace -i my-images -o ./output --cache-dir ./cache

# Later runs: try different settings without rereading the images
imagespace -i my-images -o ./output --cache-dir ./cache --relayout --min-cluster-size 50
```

## Record of settings

Every run writes the model and settings it used to `data/analysis_config.json`. The viewer shows the same record when you click the book icon.

## Testing

```bash
pip install '.[test]'
pytest                                  # quick checks, no downloads
python3 tests/run_checks.py --seed 42   # full check: two seeded runs must match
```

[`tests/README.md`](../tests/README.md) explains both.
