import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
# import matplotlib.pyplot as pltp
import mediapipe as mp
import pickle 
import time
import sys
from hog import extract_hog_features

# Function to extract bounding boxes from hand landmarks
def extract_bounding_boxes(image, landmarks):
    bounding_boxes = []
    if landmarks is not None:

        for hand_landmarks in landmarks:
            landmark_points = []
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                landmark_points.append([x, y])
            landmark_points = np.array(landmark_points)
            x, y, w, h = cv2.boundingRect(landmark_points)
            scale_factor = 1.3
            x, y, w, h = increase_bbox((x, y, w, h), scale_factor)
            # clip the values of x and y to be within the image dimensions
            x = max(0, x)
            y = max(0, y)
            bounding_boxes.append((x, y, w, h))
    return bounding_boxes

# Function to increase bounding box size
def increase_bbox(bbox, scale_factor):
    x, y, w, h = bbox
    delta_w = int((scale_factor - 1) * w / 2)
    delta_h = int((scale_factor - 1) * h / 2)
    return x - delta_w, y - delta_h, w + 2 * delta_w, h + 2 * delta_h

# Function to extract HOG features from a region of interest (ROI)
def extract_hog_from_roi(image, bbox):
    x, y, w, h = bbox
    roi = image[y:y+h, x:x+w]
    # print("roi size: ", roi.size)
    if roi.size == 0:
        print("roi size is 0")
        return None
    resized_roi = cv2.resize(roi, (64, 128))  # Resize the ROI to a fixed size
    # hog = cv2.HOGDescriptor((64, 128), (16,16), (8,8), (8,8), 9)
    # features = hog.compute(resized_roi)
    features = extract_hog_features(resized_roi, cell_size=(8, 8), block_size=(16, 16), block_stride=(8, 8), num_bins=9, win_size=(64, 128))
    return features.flatten()

# Function to load images
def load_images(images_path):
    images = []
    for image_path in images_path:
        image = cv2.imread(image_path)
        images.append(image)
    return images


def test_model_rt(image, clf):
    # Given an image and a trained classifier, predict the hand state
    # Load the image
    # img = cv2.imread(image)
    img = image
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Initialize MediaPipe Hands
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=True, 
                          max_num_hands=2,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5)

    # Detect hands and extract landmarks
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        bounding_boxes = extract_bounding_boxes(img, results.multi_hand_landmarks)

        # display the bounding boxes on the image 
        for bbox in bounding_boxes:
            x, y, w, h = bbox
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # cv2.imshow("Hand Detection", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        for bbox in bounding_boxes:
            hog_features = extract_hog_from_roi(img, bbox)
            if hog_features is not None:
                hog_features = hog_features.reshape(1, -1)
                # convert the hog features to desired form for prediction
                # hog_features = np.array(hog_features)
                prediction = clf.predict(hog_features)
                print("Prediction:", prediction)
                # convert the prediction to integer
                prediction = int(prediction[0])

                
                
                return prediction
            
        
    return None



def test_model(image_path, clf):
    # Given an image path and a trained classifier, predict the hand state
    # Load the image
    img = cv2.imread(image_path)
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # show the image
    # cv2.imshow("Image", img)
    # cv2.waitKey(0)

    
    # Initialize MediaPipe Hands
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=True, 
                          max_num_hands=2,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5)

    # Detect hands and extract landmarks
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        bounding_boxes = extract_bounding_boxes(img, results.multi_hand_landmarks)

        # display the bounding boxes on the image 
        for bbox in bounding_boxes:
            x, y, w, h = bbox
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # cv2.imshow("Hand Detection", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        output = []
        for bbox in bounding_boxes:
            hog_features = extract_hog_from_roi(img, bbox)
            if hog_features is not None:
                # print the hog features
                # print("HOG Features: ", hog_features)
                hog_features = hog_features.reshape(1, -1)
                
                # convert the hog features to desired form for prediction
                # hog_features = np.array(hog_features)
                prediction = clf.predict(hog_features)
                # convert the prediction to integer
                prediction = int(prediction[0])
                # print("Prediction:", prediction)
                # convert the prediction to integer
                # prediction = int(prediction[0])
                output.append(prediction)
        
        return output
            
        
    return None

# Main function
if __name__ == "__main__":
    # Paths to images
    # dataset_folder = "Final/closed3/train/"

    # Initialize MediaPipe Hands
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=True, 
                          max_num_hands=2,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5)
    
    # Initialize SVM classifier
    clf = svm.SVC(kernel='linear')

    # Initialize data lists
    X = []
    y = []

    dataset_folders = ["Final/closed3/train/", "Final/open1/train/", "Final/open2/train/", "Final/closed1/train/", "Final/closed2/train/"]
    

    # # TRAINING THE MODEL
    
    # start = time.time()     # time the model training process
    # for dataset_folder in dataset_folders:
            
    #     print("Processing dataset:", dataset_folder)
    #     # Process each image to generate bounding boxes and extract HOG features
    #     for filename in os.listdir(dataset_folder):
    #         if filename.endswith(".jpg"):
    #             # Load the image
    #             img_path = os.path.join(dataset_folder, filename)
    #             img = cv2.imread(img_path)
    #             img = cv2.flip(img, 1)
    #             imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #             # Detect hands and extract landmarks
    #             results = hands.process(imgRGB)

    #             # check if hands are detected
    #             if results.multi_hand_landmarks:
    #                 # Extract bounding boxes from hand landmarks
    #                 bounding_boxes = extract_bounding_boxes(img, results.multi_hand_landmarks)
    #                 # Extract HOG features from within bounding boxes
    #                 for bbox in bounding_boxes:
    #                     hog_features = extract_hog_from_roi(img, bbox)
    #                     if hog_features is not None:
    #                         X.append(hog_features)
    #                         if "closed" in dataset_folder:
    #                             y.append(0)
    #                         else:
    #                             y.append(1)

    
    
    # # Convert lists to numpy arrays
    # X = np.array(X)
    # y = np.array(y)
   
    # # Train SVM classifier
    # clf.fit(X, y)

    # # save the model
    # with open("svm_model.pkl", "wb") as f:
    #     pickle.dump(clf, f)
    
    # end = time.time()
    # print("Time taken to train the model: ", end - start, "seconds")



    # load the model from the pickle file
    with open("svm_model.pkl", "rb") as f:
        clf = pickle.load(f)
    




    # # Predict on training data 
    # # y_pred = clf.predict(X)

    # start = time.time()

    # # val_folders = ["Final/closed2/valid/", "Final/open2/valid/", "Final/closed1/valid/", "Final/open1/valid/", "Final/closed3/valid/"]
    # val_folders = ["test"]
    # # Test on the val data
    # # val_folder = "Final/closed3/valid/"
    # X_val = []
    # y_val = []
    # for val_folder in val_folders:
    #     # sort the files in the folder
    #     files = os.listdir(val_folder)
    #     files.sort()

    #     for filename in files:
    #         if filename.endswith(".jpg"):
                
    #             # Load the image
    #             img_path = os.path.join(val_folder, filename)
    #             img = cv2.imread(img_path)
    #             img = cv2.flip(img, 1)
    #             imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #             # Detect hands and extract landmarks
    #             results = hands.process(imgRGB)
    #             # print("processing image: ", filename)
    #             if results.multi_hand_landmarks:
    #                 bounding_boxes = extract_bounding_boxes(img, results.multi_hand_landmarks)
    #                 for bbox in bounding_boxes:
    #                     hog_features = extract_hog_from_roi(img, bbox)
    #                     if hog_features is not None:
    #                         X_val.append(hog_features)
    #                         if "closed" in val_folder:
    #                             y_val.append(0)
    #                         else:
    #                             y_val.append(1)

                

                
                        
    # X_val = np.array(X_val)
    # y_val = np.array(y_val)
    # y_pred = clf.predict(X_val)
    
    # print("y_val: ", y_val)
    # print("y_pred: ", y_pred)
    
    # end = time.time()
    # print("Time taken to test the model: ", end - start, "seconds")
    
    

    # # print the accuracy
    # accuracy = np.mean(y_val == y_pred)
    # print("Accuracy:", accuracy)


    # # Evaluation
    # print("Classification Report:")
    # print(classification_report(y_val, y_pred))

    # # print the false positive rate and true positive rate
    # print("False Positive Rate: ", np.sum((y_val == 0) & (y_pred == 1)) / np.sum(y_val == 0))
    # print("True Positive Rate: ", np.sum((y_val == 1) & (y_pred == 1)) / np.sum(y_val == 1))
    # print("False Negative Rate: ", np.sum((y_val == 1) & (y_pred == 0)) / np.sum(y_val == 1))
    # print("True Negative Rate: ", np.sum((y_val == 0) & (y_pred == 0)) / np.sum(y_val == 0))

    # # Plot ROC curve
    # fpr, tpr, _ = roc_curve(y_val, y_pred)
    # auc = roc_auc_score(y_val, y_pred)
    # print("AUC:", auc)
    # # print the precision and recall
    # # print("Precision: ", np.sum((y_val == 1) & (y_pred == 1)) / np.sum(y_pred == 1))
    # # print("Recall: ", np.sum((y_val == 1) & (y_pred == 1)) / np.sum(y_val == 1))
    # # print("F1 Score: ", 2 * np.sum((y_val == 1) & (y_pred == 1)) / (np.sum(y_val == 1) + np.sum(y_pred == 1)))

    # plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve')
    # plt.legend()
    # plt.show()


    # read the input_dir and output_dir from the command line
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    # output_dir = "output"



    # Predict on training data 
    # y_pred = clf.predict(X)

    # start = time.time()

    # val_folders = ["Final/closed2/valid/", "Final/open2/valid/", "Final/closed1/valid/", "Final/open1/valid/", "Final/closed3/valid/"]
    # val_folders = ["test"]
 
    # for val_folder in val_folders:
    val_folder = input_dir
    print("Reading images from", val_folder)
    # sort the files in the folder
    files = os.listdir(val_folder)
    files.sort()
    for filename in files:
        if filename.endswith(".jpg"):
            
            # Load the image
            img_path = os.path.join(val_folder, filename)
            img = cv2.imread(img_path)
            img = cv2.flip(img, 1)
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # create copy of image for output
            img_out = img.copy()
            # Detect hands and extract landmarks
            results = hands.process(imgRGB)
            # print("processing image: ", filename)
            if results.multi_hand_landmarks:
                bounding_boxes = extract_bounding_boxes(img, results.multi_hand_landmarks)
                for bbox in bounding_boxes:
                    hog_features = extract_hog_from_roi(img, bbox)
                    # predict the hand state
                    if hog_features is not None:
                        hog_features = hog_features.reshape(1, -1)
                        prediction = clf.predict(hog_features)
                        # convert the prediction to integer
                        prediction = int(prediction[0])
                        # draw the bounding box on the image
                        x, y, w, h = bbox
                        cv2.rectangle(img_out, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        # add the prediction to the image
                        if prediction == 0:
                            prediction = "Closed"
                        else:
                            prediction = "Open"
                        # cv2.putText(img_out, prediction, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                            # write at the center of the bounding box such that the text is center aligned
                        text_size = cv2.getTextSize(prediction, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                        text_x = x + (w - text_size[0]) // 2
                        text_y = y + (h + text_size[1]) // 2
                        cv2.putText(img_out, prediction, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                
                # save the output image
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, img_out)
                # print("Output image saved to", output_path)

    print("Output images saved to", output_dir)
    # end = time.time()
    # print("Time taken to test the model: ", end - start, "seconds")
    



    # start = time.time()

    # output_file = "output.txt"

    # val_folders = ["Final/closed2/valid/", "Final/open2/valid/", "Final/closed1/valid/", "Final/open1/valid/", "Final/closed3/valid/"]

    # # val_folders = ["test"]
    # outputs = []
    # for val_folder in val_folders:
    #     for filename in os.listdir(val_folder):
    #         if filename.endswith(".jpg"):

    #             # Load the image
    #             img_path = os.path.join(val_folder, filename)
                
    #             # predict the hand state
    #             output = test_model(img_path, clf)
    #             curr_output = []
    #             if output is not None:
    #                 # output the predictions in single line
    #                 for prediction in output:
    #                     if prediction == 0:
    #                         prediction = "Closed"
    #                     else:
    #                         prediction = "Open"
    #                     curr_output.append(prediction)
    #                     print(prediction, end=" ")
    #                 print()
    #             # add the filename and the predictions to the outputs list
    #             file_path = os.path.join(val_folder, filename)
    #             outputs.append([file_path] + curr_output)
    
    # end = time.time()
    # print("Time taken to test the model: ", end - start, "seconds")
            
    # # write the output to a file
    # with open(output_file, "w") as f:
    #     for output in outputs:
    #         f.write(output[0] + " " + " ".join(output[1:]) + "\n")
    #     print("Output written to", output_file)
