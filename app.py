import cv2
import numpy as np
import pygame
import time
import pickle
import sys
from main import test_model_rt

# Load pre-trained hand gesture recognition model
def load_model():
    # load the model from the pickle file
    with open("svm_model.pkl", "rb") as f:
        clf = pickle.load(f)
    return clf


# Function to process video frames and control music player
def process_frame(frame, model, actions):
    # Preprocess frame (resize, normalize, etc.)
    # Replace 'preprocess_frame' with your preprocessing function
    # processed_frame = preprocess_frame(frame)
    
    # Extract hand gesture from frame
    prediction = test_model_rt(frame, model)

    # check if prediction is None
    if prediction is None:
        print("No hand gesture detected.")
        return

    # Get action corresponding to predicted hand gesture
    action = actions[prediction]

    # print the action
    print("Detected Gesture:", action)


    # Perform action based on detected gesture
    if action == "Open Hand - Play":
        # check if music is already playing
        if pygame.mixer.music.get_busy():
            # dont do anything
            print("Music is already playing...")
            return
        # check if music is paused
        if pygame.mixer.music.get_pos() > 0:
            # unpause the music
            pygame.mixer.music.unpause()
            print("Unpausing music...")
        else:
            # start playing the music
            pygame.mixer.music.play()
            print("Playing music...")
    elif action == "Closed Hand - Pause":
        # check if music is already paused
        if pygame.mixer.music.get_busy() and pygame.mixer.music.get_pos() > 0:
            # pause the music
            pygame.mixer.music.pause()
            print("Pausing music...")
        else:
            # dont do anything
            print("Music is already paused...")
    elif action == "Thumbs Up - Stop":
        pygame.mixer.music.stop()

        # Display frame
    # cv2.imshow("Hand Gesture Music Player", frame)

# Main function to capture video from webcam and process frames
def main():

    # read the audio file and video file from command line
    audio_file = sys.argv[1]
    video_file = sys.argv[2]
    
    model = load_model()

    # Define actions corresponding to different hand gestures
    actions = {
        0: "Closed Hand - Pause",
        1: "Open Hand - Play",
        2: "Thumbs Up - Stop",
        # Add more actions as needed
    }

    # Initialize Pygame for audio playback
    pygame.init()
    pygame.mixer.init()

    # Load music file
    # music_file = "summer-night-piano-solo-6885.mp3"
    music_file = audio_file
    pygame.mixer.music.load(music_file)
    pygame.mixer.music.set_volume(0.7)
    
    
    # # Clicking a live photo on command
    # while True:
    #     print("Press 'c' to click a photo, 'q' to exit")
    #     query = input("  ")
    #     if query == 'c':
    #         cap = cv2.VideoCapture(0)
    #         ret, frame = cap.read()
    #         process_frame(frame)
    #         cap.release()
    #     elif query == 'q':
    #         break
       
       

    # # Read from a image file and process it
    # while True:
    #    print("Press 'c' to click a photo, 'q' to exit")
    #    query = input("")
    #    if query == 'c':
    #         print("Enter the path to the image file")
    #         path = input("")
    #         image = cv2.imread(path)
            
    #         process_frame(image)
    #    elif query == 'q':
    #         break

    

    # vid_file = "test_video.mp4"
    vid_file = video_file
    # ######## Run on local video file
    # # Open video file
    
    cap = cv2.VideoCapture(vid_file)

    while True:
        # Read frame from video every 1s
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame to control music player
        process_frame(frame, model, actions)

        
        # put the video reader one second forward
        cap.set(cv2.CAP_PROP_POS_MSEC, cap.get(cv2.CAP_PROP_POS_MSEC) + 400)

        # wait for 1s
        # time.sleep(1)

        # show the frame
        cv2.imshow("Hand Gesture Music Player", frame)

        # Exit if 'e' is pressed
        if cv2.waitKey(1) & 0xFF == ord('e'):
            break

    
    # #### LIVE VIDEO FEED

    # # Open webcam
    # cap = cv2.VideoCapture(0)

    # while True:
    #     # Read frame from webcam
    #     ret, frame = cap.read()

    #     if not ret:
    #         break

    #     # Process frame to control music player
    #     process_frame(frame)

    #     # Display frame
    #     cv2.imshow("Hand Gesture Music Player", frame)

    #     # Exit if 'q' is pressed
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # # Release resources
    # cap.release()
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
