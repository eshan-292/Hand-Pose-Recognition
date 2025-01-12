# 🖐️ Hand Gesture Recognition Using HOG Features  

**Author:** Eshan Jain  

---

## 🚀 Project Overview  

This project focuses on implementing a hand gesture recognition system using **HOG (Histogram of Oriented Gradients) features** and a **Support Vector Machine (SVM)** for classification. The system detects whether a hand is open or closed and extends its application to real-world scenarios, including controlling a music player and a game with hand gestures.

---

## 🛠️ Tools and Technologies Used  

- **OpenCV**: For image processing.  
- **scikit-learn**: For SVM implementation and metrics evaluation.  
- **MediaPipe**: For detecting hands and extracting bounding boxes.  
- **NumPy**: For mathematical operations.  
- **PyGame**: For developing the music player and game applications.  
- **pickle**: For saving and loading trained models.  
- **time**: For recording execution time.  

---

## 📋 Methodology  

1. **Hand Detection**:  
   - MediaPipe is used to detect hands in images and define bounding boxes.  

2. **Feature Extraction (HOG)**:  
   - Images are resized to a fixed size.
   - Gradients are calculated using the Sobel operator.  
   - Images are divided into blocks, and histograms are computed for each block.  
   - Block histograms are normalized to account for lighting and contrast variations.  

3. **Model Training**:  
   - HOG features are flattened and paired with corresponding labels (0 for closed hand, 1 for open hand).  
   - An SVM model is trained and saved for later use.  

4. **Prediction**:  
   - The trained model predicts labels for test images, with labels displayed on bounding boxes in output images.  

---

## 💡 Results  

- **Training Time**: ~7 minutes for 5–6k images.  
- **Validation Time**: ~1 minute for 800 images.  
- **Performance Metrics**:  
  - **Accuracy**: 100%  
  - **Precision**: 1  
  - **Recall**: 1  
  - **AUC (Area Under ROC Curve)**: 1  

The HOG-based feature detector outperformed deep learning approaches in terms of speed and resource efficiency, making it an excellent choice for real-world applications.

---

## 🎮 Real-World Applications  

### 1️⃣ Music Player  
- **Functionality**:  
  - Open hand: Plays or resumes the music.  
  - Closed hand: Pauses the music.  
- **Potential**: Can be extended to include features like skipping to the next or previous track with additional training.  

### 2️⃣ Game Controller  
- **Functionality**:  
  - Open hand: Moves an object forward/right.  
  - Closed hand: Moves an object backward/left.  
- **Potential**: A robust dataset can enable complex actions, making it suitable for VR and 3D games.  

---

## 🛠️ Challenges Faced  

1. **HOG Implementation**:  
   - Complex steps required careful dimension checks after each operation.  

2. **Bounding Box Coordinates**:  
   - MediaPipe sometimes returned invalid coordinates, which required adjustments to remain within image bounds.  

3. **Low-Resolution Webcam**:  
   - Real-time applications faced issues due to poor webcam resolution.  

4. **Dataset Limitations**:  
   - Dataset specificity limited the generalizability of real-time applications.  

---

## 📚 References  

- OpenCV Documentation: [OpenCV](https://docs.opencv.org/4.x/index.html)  
- scikit-learn Documentation: [scikit-learn](https://scikit-learn.org/stable/)  
- Sample Audio: [Pixabay](https://pixabay.com/sound-effects/search/samples/)  

---

## 🔧 How to Run  

1. For main.py: Run the following command:-
python3 main.py <path_to_input_dir> <path_to_output_dir>

-> path_to_input_dir: path to the directory containing the input images to be tested
-> path_to_output_dir: path to the directory where the labelled output images are to be stored 

Example Running Command:
python3 main.py test/ output/


2. For app.py: Run the following command:-
python3 app.py <path_to_audio_file> <path_to_video_file>

-> path_to_audio_file: path to the audio file to be played
-> path_to_video_file: path to the video on which inference has to be generated

Example Running Command:
python3 app.py summer-night-piano-solo-6885.mp3 test_video.mp4

3. For app2.py: Run the following command:-
python3 app2.py <path_to_video_file>
-> path_to_video_file: path to the video on which inference has to be generated

Example Running Command:
python3 app2.py test_video.mp4







File Structure:

1. main.py : contains the main code for training the model and generating inferences
2. app.py : contains code for the real world application
3. hog.py : contains the code for extracting the hog features 
4. vid_maker.py : contains the code used to generate the test_video for the real world application

