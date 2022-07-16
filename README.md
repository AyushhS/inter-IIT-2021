# BOSCH'S Age and Gender Prediction Model
This Model Predicts Age and Gender by processing Face data in a video or a static image.

* Datasets used - <a href='https://susanqq.github.io/UTKFace/'>UTKFace dataset </a>
* Uses Harcascade Pre-trained OpenCV models to detect Faces from Footage
* Uses 2 Seperate Deep CNNs to detect Age and Gender from Face Data

Architecture of the manually trained networks are present in the ```Age model plot.png``` and ```Gender model plot.png``` for Age and Gender models respectively.

Sample results of Training _(Title are the Predictions)_ -

![](output.png)  ![](output2.png)

## How to use -

Step 1 - Clone the repo to the desired Folder.

Step 2 - Install the required packages -

```bash
pip install -r requirements.txt
```
Step 3 - Run the tracker.py file, with the first argument as the input file.

```bash
python tracker.py video.mp4
```

Optional arguments - Folders containing multiple videos can also be directly processed, and ouput folder can also be specified.

```
python tracker.py VideoFolder Outputfolder
```
---
Done as an attempt for Inter IIT 2021, representing IIT Bhubaneswar. Team members - Ayush Soni (Team Leader), Ojjasv Gupta, Akash Mishra, Swapnil Gupta, Amit Kumar Pandit, Dhananjay Singh.