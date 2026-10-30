# CV_PR3 --- Computer Vision Project

**Red & White Skill Education \| Deep Learning \| PR 3**

This project is a Jupyter Notebook practical covering classical computer
vision techniques and deep-learning-based detection using OpenCV, YuNet,
and YOLOv8.

## Project Overview

The notebook is organized into five tasks:

1.  **Environment Setup, Image I/O & Morphological Operations**
    -   Image loading and grayscale conversion
    -   Erosion, dilation, opening, and closing
    -   Comparison of kernel shapes and sizes
2.  **Bitwise Operations & Image Histograms**
    -   AND, OR, XOR, and NOT operations on binary masks
    -   Region-of-interest masking
    -   Grayscale and colour-channel histograms
    -   Brightness and contrast adjustment
3.  **Face Detection with YuNet**
    -   Face detection using OpenCV's `FaceDetectorYN`
    -   Confidence scores and five facial landmarks
    -   Score-threshold comparison
    -   Face blurring/anonymisation
4.  **Object Detection with YOLOv8**
    -   YOLOv8-Nano pretrained on the COCO dataset (80 classes)
    -   Object detection on sample images
    -   Confidence and IoU threshold experiments
    -   Per-class detection summary
5.  **Integrated Pipeline & Final Comparison**
    -   YuNet face detection and YOLOv8 object detection on the same
        images
    -   Morphological pre-processing experiment
    -   FPS benchmarking
    -   Final comparison table

The last notebook cell also defines optional webcam functions for YuNet
face detection, YOLOv8 object detection, and the combined pipeline.

## Project Structure

The notebook expects the following folder structure:

``` text
CV_PR3/
├── CV_PR3.ipynb
├── data/
│   ├── images/
│   │   ├── animal_dog.jpg
│   │   ├── nature_flower.jpg
│   │   ├── person1.jpg
│   │   ├── person2.jpg
│   │   ├── person3.jpg
│   │   ├── city_street.jpg
│   │   ├── objects_indoor.jpg
│   │   └── road.jpg
│   └── models/
│       └── face_detection_yunet_2023mar.onnx
└── plots/
```

`data/models/` and `plots/` are created by the notebook when needed. The
YuNet ONNX model is downloaded by the notebook if it is not already
present. The sample images listed above must be supplied in
`data/images/`.

## Requirements

-   Python
-   Jupyter Notebook or JupyterLab
-   Webcam (only for the optional webcam section)

### Python Libraries

Install the dependencies in a terminal or notebook cell:

``` bash
pip install opencv-python==5.0.0.* opencv-contrib-python==5.0.0.* ultralytics matplotlib numpy
```

The notebook uses: - **OpenCV** --- image processing, morphology,
bitwise operations, and YuNet face detection - **NumPy** --- array and
numerical operations - **Matplotlib** --- image grids, histograms, and
plots - **Ultralytics** --- YOLOv8 object detection

## How to Run

1.  Download or clone the project files.
2.  Make sure the sample images are placed in `data/images/` using the
    filenames shown in the project structure.
3.  Open `CV_PR3.ipynb` in Jupyter Notebook or JupyterLab.
4.  Run the setup/import cell first.
5.  Run the notebook cells from top to bottom so that variables and
    models are initialized before they are used.
6.  The YuNet model weights are downloaded automatically by the
    model-download cell if missing. YOLOv8n weights are downloaded by
    Ultralytics on first use if needed.
7.  Review the generated outputs and plots in the notebook and in the
    `plots/` folder.

## Optional: Run Webcam Detection

The final notebook cell defines these functions:

``` python
webcam_face_detection(yunet_path)
webcam_yolo_detection(yolo_model)
webcam_integrated_pipeline(yunet_path, yolo_model)
```

Run the setup, YuNet model-download, and YOLOv8 model-loading cells
before calling the relevant function. Then call one function at a time.

-   `webcam_face_detection(yunet_path)` --- real-time YuNet face
    detection.
-   `webcam_yolo_detection(yolo_model)` --- real-time YOLOv8 object
    detection.
-   `webcam_integrated_pipeline(yunet_path, yolo_model)` --- combined
    face and object detection.

Press **Q** in the webcam window to stop. Webcam access depends on
operating-system camera permissions and the environment in which Jupyter
is running. A remote notebook session may not have access to your local
webcam.

## Expected Outputs

Depending on the cells executed, the notebook displays or saves:

-   Morphological operation and kernel comparison grids
-   Bitwise operation visualisation
-   Image histograms and brightness/contrast comparisons
-   YuNet face detections with confidence scores and landmarks
-   Face-blurred/anonymised images
-   YOLOv8 annotated object detections
-   Threshold experiment results and per-class summaries
-   Integrated face/object detection results
-   FPS benchmark and final comparison visualisation

Generated plots are saved in the `plots/` directory where specified by
the notebook.

## Notes

-   Run notebook cells in order; later tasks depend on variables and
    models created earlier.
-   Ensure all required sample images exist before running the
    image-processing and detection tasks.
-   Detection results depend on image quality, lighting, model
    confidence thresholds, and the selected input images.
-   The webcam functions are optional and are not run automatically.

## Technologies

-   Python
-   OpenCV
-   YuNet
-   Ultralytics YOLOv8
-   NumPy
-   Matplotlib
-   Jupyter Notebook
