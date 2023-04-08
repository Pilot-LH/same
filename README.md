# Segment Anything Model Exploration

## Table of Contents

- [Preparation](#preparation)
- [Interactive Image Segmentation](#seg)
- [Video Object Segmentation](#vos)
- [References](#references)

## Preparation <a name="preparation"></a>
- Following the guide of [Segment Anything](https://github.com/facebookresearch/segment-anything)

## Interactive Image Segmentation <a name="seg"></a>
- [ ] Building GUI with ChatGPT

## Video Object Segmentation <a name="vos"></a>

<p align="center">
  <img src="vos_demo.gif" width="640" />
</p>


1. Data in `demo/` folder is from [DAVIS](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-dev-Full-Resolution.zip). In general, the data should be in the following format:
    ```
    demo/
        video1/
            00000.png # initial masks
            00000.jpg # inital rgb frame
            00001.jpg
            ...
        video2/
            00000.png # initial masks
            00000.jpg # inital rgb frame
            00001.jpg
            ...
        ...
    ```
2. Notebook `vos_exp.ipynb` shows an example of using SAM for one-shot video object segmentation
3. Generate `vos_demo.gif` by applying the same method in `vos_exp.ipynb` to multiple frames

## References <a name="references"></a>
- Segment Anything: [[`Paper`](https://ai.facebook.com/research/publications/segment-anything/)] [[`Project`](https://segment-anything.com/)]
