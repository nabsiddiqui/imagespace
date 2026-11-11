<div align="center">
  <img src="assets/logo.svg" width="100" alt="ImageSpace logo"/>
  <h1>ImageSpace</h1>
</div>

ImageSpace turns a folder of images into a website you can explore. It places similar images near one another on a map, groups them into clusters, and lets you zoom, filter, and open any image to see the images most like it.

The website is an ordinary folder of files. It needs no database and no server-side code, so it can be put on any web host.

## 1. Try the sample

The repository includes a finished example made from 500 WikiArt images. You do not need to install ImageSpace to view it. You need only a web browser and Python 3.

Download this repository, open a terminal in its folder, and run:

```bash
python3 -m http.server 5174 -d examples/sample-output
```

Then open http://localhost:5174 in your browser. Press Ctrl+C in the terminal when you are done.

Do not open `index.html` by double-clicking it. Browsers block the files the viewer needs unless they come from a web server, so the page would be blank. The command above starts a simple web server for you.

A larger [live demo](https://nabeelsiddiqui.net/imagespace-demo) shows 49,585 WikiArt images.

## 2. Use your own images

You need Python 3.10 or newer.

**Install** (once), from the repository folder:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install '.[full]'
```

**Build the website** from a folder of images:

```bash
imagespace -i /path/to/your/images -o ./output
```

**View it:**

```bash
python3 -m http.server 5174 -d ./output
```

Then open http://localhost:5174.

The first run downloads the CLIP model, and reading the images is the slowest step. A graphics processor (NVIDIA or Apple Silicon) makes it faster but is not required.

## 3. Add information about your images (optional)

If you have a spreadsheet describing your images, save it as a CSV file with a `filename` column that matches the image file names:

```csv
filename,artist,style,title
monet_water-lilies.jpg,Claude Monet,Impressionism,Water Lilies
picasso_guernica.jpg,Pablo Picasso,Cubism,Guernica
```

Then pass it to ImageSpace:

```bash
imagespace -i /path/to/your/images -o ./output --metadata metadata.csv --label-column style
```

Columns such as `style` become filters in the viewer. With `--label-column`, each cluster also shows the most common values in that column, so you can see what the cluster holds.

To try this with the bundled sample images:

```bash
imagespace -i test-data -o ./output --metadata test-data/metadata.csv --label-column style --seed 42
```

## 4. Share the website

The `output` folder holds the whole website: `index.html` (the page to open), `assets/` (the viewer), and `data/` (your images, the map, and a record of the settings used).

Upload that folder to any static web host, such as GitHub Pages, Netlify, or Cloudflare Pages. Nothing needs to be installed on the host.

## What you will see

- **Five views** of the same images: the similarity map (UMAP), the map with overlaps removed (Snap), and Grid, Color, and Timeline.
- **Hotspots:** the clusters, numbered by size. Click one to see only its images and, if you supplied a metadata column, what it holds.
- **Noise:** images that fit no cluster are hidden at first. The Noise control shows them.
- **Detail panel:** click any image to see its information and the ten images most like it.
- **Methods & Provenance** (book icon): the model and settings used to build the site.

## Platform notes

- **Mac with Apple Silicon:** the graphics processor is used automatically.
- **Windows or Linux with an NVIDIA card:** install the CUDA versions of PyTorch and `onnxruntime-gpu` to use it.
- **No graphics processor:** everything works, but reading the images takes longer.
- **Browser:** use a current version of Chrome, Edge, Firefox, or Safari. Phones show lower-resolution images to save memory.

## Troubleshooting

- **Blank page:** you opened `index.html` directly. Start the web server with `python3 -m http.server` as shown above.
- **`CLIP embedding extraction failed`:** install with `pip install '.[full]'`.
- **`--label-column` error:** the column must exist in your metadata CSV, and the `filename` column must match the image file names.
- **No clusters found:** small or very mixed collections may not form clusters. Try a lower `--min-cluster-size`.
- **Old results after changing the model or images:** use a new `--cache-dir`.

## More detail

[`docs/reference.md`](docs/reference.md) lists every setting you can change, including how to make reruns faster. [`tests/README.md`](tests/README.md) describes the tests.

## License

MIT
