# Dataset Folder

Place your ear image dataset here. Each **sub-folder** represents one person (class).

## Expected Layout

```
dataset/
    Person_01/
        ear_001.jpg
        ear_002.jpg
        ear_003.jpg
        ...
    Person_02/
        ear_001.jpg
        ear_002.jpg
        ...
    Person_03/
        ...
```

## Tips
- Minimum **5 images per person** for decent accuracy.
- Images can be JPG, PNG, BMP, PGM or TIFF.
- Sub-folder names become the displayed identity labels.
- You can use the AWE (Annotated Web Ears) dataset or your own captured images.
- Enable **Augmentation** in the Train tab to multiply your dataset 6× automatically.

## AWE Dataset
Download from: https://awe.fri.uni-lj.si/
Extract and point the Train tab to the root folder containing one folder per subject.
