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

# Function to process video frames and control game object movement
def process_frame(frame, model):
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
    action = prediction

    # print the action
    print("Detected Gesture:", action)

    # Perform action based on detected gesture
    if action == 0:  # Closed hand gesture moves object left
        move_object(-3, 0)
    elif action == 1:  # Open hand gesture moves object right
        move_object(3, 0)

# Function to move the game object
def move_object(dx, dy):
    global object_position
    object_position[0] += dx
    object_position[1] += dy

# Main function to capture video from webcam and process frames
def main():
    model = load_model()
    video_file = sys.argv[1]


    # Initialize Pygame
    pygame.init()

    # Set up screen
    screen_width = 800
    screen_height = 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Hand Gesture Game")

    # Set up colors
    BLUE = (0, 0, 255)
    WHITE = (255, 255, 255)

    # Initialize game object
    object_width = 50
    object_height = 50
    global object_position
    object_position = [screen_width // 2 - object_width // 2, screen_height // 2 - object_height // 2]
    vid_file = video_file
    cap = cv2.VideoCapture(vid_file)
    # Game loop
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Capture frame from webcam
        
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame to control game object movement
        process_frame(frame, model)

        # put the video reader one second forward
        cap.set(cv2.CAP_PROP_POS_MSEC, cap.get(cv2.CAP_PROP_POS_MSEC) + 400)

        # show the frame
        cv2.imshow("Hand Gesture Music Player", frame)

        # Draw background
        screen.fill(WHITE)

        # Draw game object
        pygame.draw.rect(screen, BLUE, [object_position[0], object_position[1], object_width, object_height])

        # Update display
        pygame.display.flip()
        # sleep for 1 seconds
        # time.sleep(1)


    # Quit Pygame
    pygame.quit()
    cap.release()

if __name__ == "__main__":
    main()
