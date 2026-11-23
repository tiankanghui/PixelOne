[← PixelOne](../README.md) · [Documentation](README.md)

# Dataset configuration

[`configs/dataset_mixtures.json`](../configs/dataset_mixtures.json) records the paper's target counts and source mixture, including the controlled 21,557 targets, separate 200K/500K continuations, and 2,000-target validation set. The exact annotations, pinned upstream download links, and image reconstruction script are hosted separately in [PixelOne-Data](https://huggingface.co/datasets/KangHui2200/PixelOne-Data). The package contains instructions and original-pixel boxes, but no screenshots. Access requires authorization while the dataset repository is private. Obtain each image source under its own license and terms.

## Annotation schema

Each JSONL record describes one grounding target:

```json
{"id":"example-0001","image":"images/example.png","instruction":"Click Search","bbox_xyxy":[120,80,180,110]}
```

`bbox_xyxy` is `[left, top, right, bottom]` in original screenshot pixels, not normalized coordinates. One screenshot may supply several targets, each with a distinct ID. The image path is relative to `image_root`; a single-element image list is also accepted. This is a grounding annotation format, not a chat/SFT message format.

```bash
pixelone prepare-data --input annotations.jsonl --image-root data \
  --output data/control/train.jsonl
```

Preparation validates images, IDs, instructions, box geometry and representable pixel centers, and adds `image_size` and `image_sha256`. It keeps the input image paths; point `image_root` at the same source image tree during training. Output files are not silently overwritten.

## Splits and screening

- Treat each source screenshot and all derived crops as one image family when splitting.
- Keep validation image families separate from both continuations and the controlled set.
- Remove benchmark-overlapping images before sampling, using decoded-RGB identity and the documented near-duplicate thresholds.
- Retain one instruction per screenshot–box pair in a dataset; exclude duplicate target IDs and screenshot–box pairs when extending 200K to 500K.

The mixture file documents these rules and thresholds. `prepare-data` validates the annotation schema; it does **not** perform cross-dataset near-duplicate screening or resample the original training sets. Use the exact annotations and reconstruction instructions in PixelOne-Data to recover the selected targets and images.

Raw datasets, personal screenshots, local image paths, benchmark predictions and cached teacher traces are not included in the public code tree or model packages.
