# AI Skin Analysis Project

## Overview
A real-time skin analysis system that uses computer vision to detect skin conditions like dryness, acne, and irritation using your webcam.

## Features
- Real-time face detection using Haar Cascade
- Skin condition analysis (dryness, acne, irritation)
- Live camera feed with face tracking
- Color-coded results (Green: Healthy, Yellow/Orange/Red: Issues)
- Dynamic thresholding for different lighting conditions

## Installation

1. **Install required packages:**

   pip install -r requirements.txt
   \\\

2. **Run the application:**
   
   python smart_mirror_skin_check.py
   \\\

## Requirements
The project requires the following packages (already in requirements.txt):
- opencv-python==4.8.1.78
- numpy==1.24.3

## Usage
1. Ensure good, even lighting on your face
2. Position yourself about 1-2 feet from the camera
3. The system will automatically detect your face
4. Results will show in real-time with color codes:
   - 🟢 Green: Healthy skin
   - 🟡 Yellow: Minor irritation
   - 🟠 Orange: Possible acne/oily skin
   - 🔴 Red: Dry skin detected

5. Press **ESC** to exit the application

## How It Works
1. **Face Detection**: Uses Haar Cascade classifier to detect faces
2. **Skin Analysis**: Converts to HSV color space and analyzes brightness/texture
3. **Condition Detection**: Uses adaptive thresholding to identify problem areas
4. **Results Display**: Shows real-time analysis with visual feedback

## Technologies Used
- Python
- OpenCV (Computer Vision)
- NumPy (Mathematical operations)
- Haar Cascade Classifier (Face detection)

## Project Structure
ai-skin-analysis/
├── smart_mirror_skin_check.py  # Main application
├── requirements.txt            # Dependencies
└── README.md                  # Project documentation

## Future Enhancements
- Add more skin condition detection
- Implement skin hydration analysis
- Add historical tracking of skin health
- Mobile app version


