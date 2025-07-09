# **AI-Powered Assistive System for Visually & Hearing Impaired**

## HackOrbit 2025 | Team VegaVerse
A real-time intelligent system that empowers the visually impaired to navigate with voice assistance and enables speech for deaf-mute users via hand gestures.

## Problem Statement
Millions of people with visual or hearing/speech disabilities struggle with daily communication and navigation. We aim to solve this with an AI-based assistive system that works offline, in real time, and on portable hardware.

## Our Solution
We developed a unified AI-powered assistive system with three core capabilities:

### For Visually Impaired:
-Object detection using a mobile/webcam camera.
-Real-time voice feedback using Text-to-Speech (TTS).

###For Deaf & Mute Users:
-Hand gesture recognition (using CNN + MediaPipe).
-Converts gestures into text + voice output.

### For Communicating to Deaf Users:
-Text-to-sign language video generation.
-Displays hand-sign avatar videos for better comprehension.

## Tech Stack

| Component           | Technology                                  |
| ------------------- | ------------------------------------------- |
| Object Detection    | YOLO / SSD                                  |
| Gesture Recognition | CNN + MediaPipe                             |
| Speech Synthesis    | gTTS / pyttsx3                              |
| Sign Language Video | Sign Generation Model + Avatar Animation    |
| Frameworks          | Python, OpenCV, TensorFlow/Keras, Streamlit |
| Deployment          | Local (Offline Capable), Portable Devices   |


## System Architecture

### Entire WorkFlow
![WorkFLow](https://github.com/user-attachments/assets/b1bb98c1-b0c2-47c4-8a3f-e822fce0e60d)

### Blind people architecture 
![Blind User Flow](https://github.com/user-attachments/assets/4efdf341-c000-4a0c-a7e4-75f39ce1dcda)


### Deaf people architecture 
![Deaf-Mute User Flow](https://github.com/user-attachments/assets/34970033-9c3a-49ac-b2b8-620dc0704e2e)


### Dumb people architecture 
![Text-to-Sign Communication](https://github.com/user-attachments/assets/74f74525-458a-422a-9bcd-69fe54063ad9)



## Features & Novelty

### Key Features
-Real-time object detection and gesture recognition
-Gesture → Text → Speech pipeline
-Text → Sign video for Deaf communication
-Offline, low-latency, portable implementation

### Novelty
-Unified system for multiple disabilities
-Combines Computer Vision + NLP
-Gesture-to-speech for inclusive communication
-Designed for rural/low-resource settings


 ## Limitations & Mitigations
 | Limitation                     | Mitigation                           |
| ------------------------------ | ------------------------------------ |
| Low lighting reduces detection | Use IR or better low-light models    |
| Limited gesture vocabulary     | Gradually expandable                 |
| Audio output in noisy areas    | Add vibration/LED feedback as backup |

##  How to Run the Project
### Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

### Install dependencies
pip install -r requirements.txt

### Run the app
streamlit run app.py

# Team VegaVerse

| Name                    | Email                                                               | Role                            |
| ----------------------- | ------------------------------------------------------------------- | ------------------------------- |
| **Madhulatha Seerapu**      | [madhuseerapu79@gmail.com](mailto:madhuseerapu79@gmail.com)         | AI & CV                         |
| **Sudharshan Paul Ganta**   | [gantasudarshanpaul@gmail.com](mailto:gantasudarshanpaul@gmail.com) | UI/Backend                      |
| **Sree Harsha Alapati** | [sreeharshaalapati@gmail.com](mailto:sreeharshaalapati@gmail.com)   | Gesture Models, TTS Integration |

## Acknowledgments
-HackOrbit 2025 Organizers
-Open-source contributors (MediaPipe, YOLO, Sign Language datasets)
