# User instructions
## Configuration file
In the folder source there is a file named `configuration.py`, this file includes all the parameters and options that can be selected by the user. This is where the relative filepath of the input video and calibration video must be defined. To use the sample videos in this project, the default paths in this file can be used.

## Instructions for use
The program requires some user interaction. The program followa this structure:
1. A window will show up displaying a reference frame from the video.
2. The user must select the four corners of the goal region of the court. This is done by pressing left-clicking in the picture. When all corners are selected the user must press `enter`.
3. A window will show up displaying the same reference frame.
4. The user must select corners in an arbitrary polygon, masking out the area of interest. This should be only the are where a balls are expected to be. Selection is done with left-click. When all corners are selected the user must press `enter`.
5. A window will show up displaying the tracked corners in the calibration video. No user action is needed.
6. A window will show up displaying the intermediate results of the ball tracking for each frame of the video. No user action is needed, but enter kan be pressed to stop the tracking at any point (in which case the following sections would only use the tracked positions up until that point).
7. A window will showing a digital, animated recreation of the game, with estimated scores. This window must be closed by the user to continue.
8. A window will show up showing the tracked trajectory of one ball. The user must select the moment where the ball leaves the hand and then the moment when the ball hits the ground. The trajectory can be traversed by pressing `n` for forward or `b` for backward. The moments are captured by pressing `c`. The moments must be selected IN ORDER. When both moments are selected the user must press `enter`. If no moments are selected before `enter` is hit, the trajectory will be disregarded in later steps.
9. A new window with the next ball trajectory will show up. Step 7 must be performed for all balls.
10. A window will show up displaying the estimated 3D-trajectories of the trajectory segments selected by the user. When this window is closed the program will terminate.

For each step where user interaction is needed there will be instructions in the upper left corner of the windows. No information or action has to be done in the terminal, everything should happen within the active window.

The user can select additional bounces after the ball hits the ground the first time. This is simply done by adding the moments where the ball hits the ground. This requires that there are enough detected points within that bounce.