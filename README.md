# Stereovision Independent Project
## Final product:
![pc-video_yqmUAoVl-ezgif com-video-to-gif-converter](https://github.com/EmmaRYoung/StereovisionProject/assets/67296859/90ec8a18-c4d9-4bcd-85f9-821acb6d51c9)
### Background 
What is stereovision? Computer stereo vision uses two or more cameras to capture and analyze images to percieve depth in a scene. 
![image](https://github.com/EmmaRYoung/StereovisionProject/assets/67296859/91af947f-f3af-4ec3-b2f9-c75f785effee)

### Method
1. Set up web cameras close to eachother and capture two images simultaneously
   
![image](https://github.com/EmmaRYoung/StereovisionProject/assets/67296859/0956b31a-9cee-4646-a76f-6c0f34dd8cc1)

3. Calculate the intrinsics of your cameras using OpenCV's chessboard calibration
   
![image](https://github.com/EmmaRYoung/StereovisionProject/assets/67296859/dcbd0c73-f118-4c89-8725-b91a97e7e044)

5. Use SIFT to identify and match key features of the image
   
![image](https://github.com/EmmaRYoung/StereovisionProject/assets/67296859/8b3ee14d-ef77-4bba-a5f9-c325f8f7ce25)

7. Rectify the two images
   
![image](https://github.com/EmmaRYoung/StereovisionProject/assets/67296859/582c271f-4f20-44f9-b957-d232cc98bca9)

9. Calculate a disparity between the two rectified images
   
![image](https://github.com/EmmaRYoung/StereovisionProject/assets/67296859/64c9ae62-10e9-48d3-a482-9a743ab7c745)

7. Using OpenCV and camera instrinsics, we can recover a point cloud of the scene!
   
![pc-video_yqmUAoVl-ezgif com-video-to-gif-converter](https://github.com/EmmaRYoung/StereovisionProject/assets/67296859/90ec8a18-c4d9-4bcd-85f9-821acb6d51c9)

