# 🎭 Real-Time Facial Accessories Integration  

**Real-Time Facial Accessories Integration** is a **computer vision-based project** that applies real-time filters like **hats, mustaches, glasses, and beards** onto detected faces using **Dlib's Haar Cascade Classifier**. Inspired by **Snapchat filters**, the system accurately detects **facial landmarks** and overlays accessories while ensuring real-time performance.  

## 🚀 Features  

- 🖼 **Real-Time Facial Feature Detection** – Identifies key facial landmarks using **Dlib**.  
- 🎭 **Filter Integration** – Overlays accessories like hats, glasses, beards, and mustaches.  
- ⚡ **Live Webcam Processing** – Applies filters dynamically using **OpenCV**.  
- 🎯 **Improved Face Detection** – More robust than OpenCV’s Haar Cascade, handling pose variations better.  
- 🏎 **Optimized Performance** – Downscaling and upscaling methods improve real-time responsiveness.  
- 🔍 **GUI for Interaction** – Users can toggle between different filters using a **Tkinter-based interface**.  

## 🛠️ Tech Stack  

- **Programming Languages:** Python  
- **Computer Vision:** OpenCV, Dlib  
- **Facial Landmark Detection:** Dlib’s 68 facial key points model  
- **GUI Framework:** Tkinter  
- **Performance Optimization:** Image downscaling & upscaling  

## 🎯 Challenges & Solutions  

- **Pose Detection Issues:** OpenCV struggled with tilted faces, so **Dlib’s Haar Cascade** was used for better robustness.  
- **Performance Bottlenecks:** Used **image downscaling and upscaling** to improve processing speed.  
- **Beard Filter Alignment Issues:** Scaling adjustments attempted, but further improvements needed.   
