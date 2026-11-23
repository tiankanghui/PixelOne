[← PixelOne](../README.md#demos)

# Browser demos

Two native-resolution recordings show the released PixelOne-Pro models grounding instructions in real browser interfaces.

## 4K spreadsheet

**PixelOne-Pro-4B** performs four verified clicks in Jspreadsheet CE: check a **12 × 13-pixel checkbox**, select a project cell, apply bold formatting, and open the team forecast.

[New original 4K MP4](assets/demos/spreadsheet-4k.mp4) · [Recording measurements](assets/demos/browser-demo-results.json)

Model inputs and the recording are **3840 × 2160**. The 15.80-second video retains its original speed and uses the same pointer, click halo and timing panel as Three Suns. The workbook contains 48 distinct fictional initiatives with individual owners, dates, budgets and delivery notes. Forecast aggregates the figures across ten teams instead of copying the project rows.

## Three Suns at 2K

**PixelOne-Pro-35B-A3B** completes **8/8 clicks** in an original browser game, from observing the sky to reviving a civilization.

[Watch inline](https://github.com/user-attachments/assets/60983252-5735-49be-a389-488aa5f106de) · [Original English MP4](assets/demos/three-suns-en-2k.mp4)

Both English and Chinese recordings use native **2560 × 1440** screenshots and video frames, with 3,600 visual tokens per request. They use different interface layouts.

| Recording | Layout seed | Successful clicks | Duration | Mean model inference | Mean screenshot-to-click time |
| :--- | ---: | ---: | ---: | ---: | ---: |
| English | 8 | 8/8 | 35.64 s | 511 ms | 1.78 s |
| Chinese | 2026 | 8/8 | 33.76 s | 472 ms | 1.71 s |

[Measurement records and original video checksums](assets/demos/three-suns-2k-results.json)

## Recording settings

The released models run on an NVIDIA H200 with native FLA and causal-conv1d kernels. Each step pairs a fresh screenshot with a supplied instruction and executes the predicted click without coordinate correction. PixelOne predicts coordinates; the demo controller supplies the task sequence.

Recordings retain their original speed. Display overlays are hidden from model inputs and appear after the actual click. The timings shown in the game videos measure server-side inference. Screenshot-to-click time also includes capture, transport and browser execution.

These selected demonstrations are separate from the paper’s benchmark results. Local checks cover screenshot dimensions, predictions and application state transitions.

Applications include the original Three Suns game, [Jspreadsheet CE](https://github.com/jspreadsheet/ce) 5.0.4, [SVG-Edit](https://github.com/SVG-Edit/svgedit) 7.4.2 and [Wikipedia](https://www.wikipedia.org/). Three Suns uses original procedural artwork.
