# 📌 VIA Annotator Extension for Visual Image Grounding

This project is an extension of the [VGG Image Annotator (VIA)](https://www.robots.ox.ac.uk/~vgg/software/via/) V2 tailored for Visual Image Grounding tasks. It supports panoptic segmentation annotations with grounding labels and captions. All the logic is contained in via.html and flask_server.py. It's tailored to reannotate GLaMM PSG images but if reference to those annotations are removed can be used for any panoptic segmentation task.


## 🚀 Getting Started
### 🔧 Set Up the Environment
```bash
conda create --name via_flask_env python=3.9
conda activate via_flask_env

# Install dependencies
conda install -c anaconda numpy pillow matplotlib
pip install pycocotools Flask flask-cors
```
### 📁 Required Files
All required resources are included in the original ZIP file (`glamm_annotation_data.zip`). 
You can download the full dataset from the shared location (e.g., Google Drive): [Download from Google Drive](https://drive.google.com/file/d/14lP7lapSbMm-vFQuJR21VdozEdLhICjT/view?usp=sharing)  

> The file `glamm_annotation_data.zip` contains the complete `data/` folder and must be **unzipped** (in 'via-2.0.12-vl') before use.

```bash
cd via-2.0.12-vl
unzip glamm_annotation_data.zip
```
After unzipping, you should have the following structure:

```text
data/
├── glamm_images/                      # GLaMM test and validation original images 
│   └── val_test/
│       ├── subset_0/
│       ├── subset_1/
│       ├── subset_2/
│       ├── subset_3/
│       ├── subset_4/
│       └── subset_5/
├── glamm_annotations/      
    ├──test_gcg_coco_caption_gt.json   # GLaMM test captions and labels (JSON)
    └──val_gcg_coco_caption_gt.json    # GLaMM validation captions and labels (JSON)
└── coco_pan_annotations/
    ├── panoptic_train2017/     # COCO panoptic masks (PNG)
    └── panoptic_train2017.json # COCO panoptic annotations (JSON)
```

To manage annotations more easily and avoid duplication, the val_test set of GLaMM images has been evenly divided into 6 non-overlapping subsets. 
Each subset contains a unique portion of the dataset. No image appears in more than one subset.

## ▶️ How to Run
### Step 1: Start the Static File Server (Terminal 1)
```bash
conda activate via_flask_env
python3 -m http.server 8001
```
or 
```bash
conda activate via_flask_env && python3 -m http.server 8001
```

### Step 2: Start the Flask Server (Terminal 2 - Python Backend)
```bash
conda activate via_flask_env
python flask_server.py
```
or 
```bash
conda activate via_flask_env && python flask_server.py
```

### Step 3: Open the Annotator in Your Browser

Visit:
```
http://localhost:8001/via.html
```
> ⚠️ **Note:** Do not open `via.html` by double-clicking it in your file manager — it might fail, better accessed through the browser via the URL above.

## 📝 How to Annotate

1.  **Select a Subset:** Choose one of the GLaMM image subsets to annotate (e.g., `subset_2`).

2.  **Prepare Images:**
    *   Copy or move all images from your selected subset folder into the main `val_test` directory:
        > `cp data/glamm_images/val_test/subset_i/* data/glamm_images/val_test/`
    *   If needed, replace any existing files in the `val_test/` directory.

    > ⚠️ **Important:** Do **not** attempt to annotate from multiple subsets simultaneously. Always ensure only one subset's images are present in `data/glamm_images/val_test/` at a time. This is for compatibility with other annotators. 

3.  **Start Annotation Tool:** Launch the annotation tool as per its specific instructions (details above).

4.  **Annotation Process:**
    *   **Select an Image:** Choose an image from the `data/glamm_images/val_test/` folder within the tool.
    *   **Preloaded Annotations:** The original GLaMM annotations (caption, labels, and mask) will be automatically preloaded for the selected image.
    *   **Check Annotation Completeness:**
        *   Be aware that COCO panoptic annotations might be incomplete for some samples.
        *   If critical annotation components are missing, it's best to **drop the sample** for now.
        *   🔒 *Note: Currently, labels and masks cannot be modified within the tool.*
    *   **Captioning Requirements:**
        *   Your new caption **must mention all preloaded labels**.
        *   An error will be displayed if any preloaded labels are missing from your caption.
    *   **Handling Previously Captioned Images:**
        *   The tool will show a warning if you load an image that has already been captioned.
        *   It will ask for confirmation before allowing you to overwrite existing captions.
    *   **Useful to Know:**
        *   Labels can be selected and de-selected (double click) to show the single masks.  
        *   The Panoptic Overlay panel can be resized using the top arrows. 

## 💾 Output

Annotations are saved as `.json` files in the `captions/` folder and include:
*   Caption
*   Referenced labels
*   Masks in RLE format

TODOs 
- [ ] Allow modification of labels
- [ ] Improve user-friendlyness
- [ ] Clean prints

## 📚 Based On

This project is an extension of the [VGG Image Annotator (VIA)](https://www.robots.ox.ac.uk/~vgg/software/via/) developed by the [Visual Geometry Group](http://www.robots.ox.ac.uk/~vgg/) at the University of Oxford. Learn more and explore their official [GitLab repository](https://gitlab.com/vgg/via).

```bibtex

@inproceedings{dutta2019vgg,
  author = {Dutta, Abhishek and Zisserman, Andrew},
  title = {The {VIA} Annotation Software for Images, Audio and Video},
  booktitle = {Proceedings of the 27th ACM International Conference on Multimedia},
  series = {MM '19},
  year = {2019},
  isbn = {978-1-4503-6889-6/19/10},
  location = {Nice, France},
  numpages = {4},
  url = {https://doi.org/10.1145/3343031.3350535},
  doi = {10.1145/3343031.3350535},
  publisher = {ACM},
  address = {New York, NY, USA},
} 

@misc{dutta2016via,
  author = "Dutta, A. and Gupta, A. and Zissermann, A.",
  title = "{VGG} Image Annotator ({VIA})",
  year = "2016",
  howpublished = "http://www.robots.ox.ac.uk/~vgg/software/via/",
  note = "Version: X.Y.Z, Accessed: INSERT_DATE_HERE" 
}
```
