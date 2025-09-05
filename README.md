# POKEMON CARD DETECTOR
Developed a real-time Pokémon card stat reader using a custom-trained YOLOv8n object detection model and OCR. The goal is to pull live data from the web using https://pokemontcg.io/ to pull card pricing, Name, set_ID, and Card_Number to uniquely identify each card scanned.<br>

# HARDWARE REQUIRED

| SONYIMX500 AI CAMERA                                  | RASPBERRYPI                                         | PC-Need GPU                                         |
|------------------------------------------------------|----------------------------------------------------|----------------------------------------------------|
| <p align="center"><img src="https://github.com/user-attachments/assets/6fd3e230-1060-40df-9afe-a44479e2307a" width="150" /></p> | <p align="center"><img src="https://github.com/user-attachments/assets/2d0179f2-2d01-4506-ba50-1d7881d8f763" width="150" /></p> | <p align="center"><img src="https://github.com/user-attachments/assets/89d65f65-0520-44da-ac6b-a45f535ba4b2" width="150" /></p> |
| Link: https://www.raspberrypi.com/products/ai-camera/ | Link: https://www.raspberrypi.com/products/raspberry-pi-5/                                        | Link: NA, just need a GPU to train YOLO Model                                         |
| Price: [$108CAD](https://www.digikey.ca/en/products/detail/raspberry-pi/SC1174/24627137?gclsrc=aw.ds&gad_source=1&gad_campaignid=20282404290&gbraid=0AAAAADrbLliyfXQEUVvMYjKzIqkJOOA5I&gclid=CjwKCAjwhuHEBhBHEiwAZrvdcu4O0hh-xE3bAOjFPrEofyCMKrRmB_MbRjcnApL76BPx21bD2Bq61BoCsL8QAvD_BwE) | Price: [$85CAD](https://www.digikey.ca/en/products/detail/raspberry-pi/SC1431/21658261?gclsrc=aw.ds&gad_source=1&gad_campaignid=20291741917&gbraid=0AAAAADrbLliwML6BanXAlSYS3eoqKNexG&gclid=CjwKCAjwhuHEBhBHEiwAZrvdcnd4SGfxmmOiQ56PPaIUpkEzczryBMtFqdYtGjf07A5ui0cgBur_YhoCr8sQAvD_BwE)                                         | Price: Doesnt have to be fancy                                        |
1. If it's your first time setting up the Pi, you will need a microSD card, USB-C power supply, and micro HDMI cable to connect the Pi to a monitor.
2. If you are creating your own dataset, you will need a camera and a bunch of Pokémon cards.
3. (*Optional but recommended*) A case for your Pi and AI camera. You can find public Raspberry Pi case files online, but I have linked an [STL file](https://github.com/TippedBucket/Pokemon_card_detector-/tree/main/protective_case_files) to print the camera case and an [f3D file](https://github.com/TippedBucket/Pokemon_card_detector-/tree/main/protective_case_files) for easy alterations.<br><br>


# Example Pictures
<img width="394" height="525" alt="card_detection" src="https://github.com/user-attachments/assets/3a2ae39b-00fc-472c-bee7-e756908ba109" />
<img width="394" height="525" alt="OCR Live Data Pull" src="https://github.com/user-attachments/assets/42c90493-320e-45e0-a823-e6d3a2ec4fdc" />



# MODEL WORKFLOW<br>
![CV_PFD](https://github.com/user-attachments/assets/fbe7533c-9856-40df-8ae3-682c2784a045) <br>

# CREATING MODEL & RUNNING ON PI 
1. Gathered Pokémon card datasets from Kaggle [Example]( https://www.kaggle.com/datasets/bzhbzh35/pokemon-cards-per-set)<br>
2. Annotated data on Roboflow, labeling for Set_ID, Name, Card_Number, and the Card.
   - Augmentation steps include:
      - 90° Rotate: Clockwise, Counter-Clockwise
      - Shear: ±9° Horizontal, ±10° Vertical
      - Noise: Up to 0.1% of pixels
   - See annotated dataset [Here]( https://universe.roboflow.com/invention-time/pokemon-card-detection-haiqn)
   - ***You can test out the RF-DETR real-time object detection model*** [Here](https://app.roboflow.com/invention-time/pokemon-card-detection-haiqn/models/pokemon-card-detection-haiqn/3) for a fun example. (Must have Roboflow account, it's free, dont worry)
3. Extracted the dataset for the YOLOV8 model from Roboflow
   - This gives us a training, validation, and testing image and labeling data split for all images and labels, as well as a YAML file that contains the number of classes, class names, and local location of the data splits.
     - You can see the data_example folder to see an image and coordinate pairing for demonstration. For a normal bounding box, the first two coordinates are the x and y center of the box relative to the grid cell, and the last two are the height and width of the bounding box.
   - Additionally, given the small memory size of the IMX500 [8MB](https://developer.sony.com/imx500), we will need to quantize and compress this model into a YOLOV8N (smallest YOLOV8) model using Ultralytic.
4. Extracted a YOLOV8n model from Ultralytics
   - Thankfully, Ultralytics has a demo on how to export a YOLO model to the IMX500 Camera. In preparation, we will need to create a virtual environment and install WSL for Windows.
   - Install WSL in your ***Administrator instance of Powershell***:  ```wsl --install```
     - Make sure to reboot your PC after installation
     - Install python and dependent packages: ```sudo apt install python3-pip python3-opencv python3-venv git libatlas-base-dev```
   - Now create a virtual environment:
      ```bash
      python3 -m venv cam-env
      source cam-env/bin/activate
      pip3 install --upgrade pip
   - Now install Ultralytics in the cam-env venv: ```pip install ultralytics```
   - If everything was done correctly, you can now create the YOLOV8n model with the yaml file created earlier:
     ```bash
        yolo detect train\
           model=yolov8n.pt\
           data = path/to/your/file.yaml\
           name = pokemon_card_detection
           epochs =20 #How many passes through the training dataset
           device =0 #0 correlates to the GPU````
     
   - This will take some time to train and will depend on your GPU. It took about an hour on an old 4GB VRAM Nvidia card from a former gaming laptop I had lying around. 
5. Exported the best PyTorch model in the IMX format for the SONY IMX500. The best PyTorch model will be in the runs/weights folder created in step 4.
   - However, there's a glitch right now (or at the time of creating this tool) where the Ultralytics export to IMX feature won't work unless we move the best.pt file into the folder where we are running the yolo_train.py script, and point to it in our excecution.
   - Run the following to export the model to imx:
      ```bash
      python yolo_train.py \
         --init_model runs/path/to/best.pt \
         --export_format imx \
         --export_only
         --int8_weights
   - After about 3-5 minutes, you will see a best_imx_model folder created that will contain the packet_out.zip and the labels.txt files that we will need to get onto our Raspberry Pi.
6. On the Raspberry Pi, we will need to git clone ```picamera2``` from Raspberry Pi as well as install ```imx500_tools imx500-all```
      ```bash
      sudo apt install imx500-all imx500-tools
      git clone -b next https://github.com/raspberrypi/picamera2
      cd picamera2
      pip install -e .  --break-system-packages

7. After pointing to the picamera2 examples for the IMX500:```/Documents/picamera2/examples/imx500```, you can then run the following script to detect the card name and card ID of a card!
   ```bash
   python imx500_object_detection_demo.py --model /home/<user>/Documents/<file.rpk> --labels /home/<user>/Documents/<labels.txt> --fps 25 --bbox-normalization --ignore-dash-labels --bbox-order xy
8. It was at this point that I realized the quantization and initial model dataset size was a bit larger than expected, and took the full 8MB of memory on the SONYIMX500. So the model doesn't work as accurately as I was expecting due to the size limitation. My best guess is that it cut off some of the dataset to account for the small memory space, which lowered overall accuracy and detection.  I am still in the process of increasing the accuracy on the IMX500 camera, but in the meantime, the next section will walk you through how to do the API call for a device with more memory

# RUNNING ON DEVICE

Once you have created a YOLO model following the previous steps, you can run the model very easily on your own device, provided you have a camera. The steps are as follows:<br>
1. Install Python and set up the virtual environment (venv). I used Python 3.11 when I ran the model
   - Activate your venv, in my case, I called the venv Pokemon
     ```bash
     python -m venv pokemon
     pokemon\Scripts\activate
2. Install the dependencies:
   ```bash
   pip install opencv-python pillow pytesseract ultralytics requests tk
3. Get the Pokémon API key by signing up at [Pokemontcg.io](https://pokemontcg.io/)
4. Create your YOLO model or use the model in this repository [here](https://github.com/TippedBucket/Pokemon_card_detector-/tree/main/model)
5. Download the [Python script](https://github.com/TippedBucket/Pokemon_card_detector-/tree/main/execution_script) that runs the model and performs tesseract OCR.
   - Functionality of the script:
     - Press [ESC] to exit the program at any time
     - Press [SPACE] to take a picture of the current video frame and do inference and OCR on the frame
     - Press [P] to pick an existing file from your computer to do inference and OCR on. I implemented this because the camera on my device is of poor quality and was too blurry to do OCR on.
6. Run the script, and make sure you have activated your virtual environment and cd'd to the file directory: 
   ```bash
      python <name of file>.py
# NOTES
   - Once the program does inference on the picture, it will give you a preview of what it saw with bounding boxes, it will then crop the bounding boxes for OCR, and take the name an ID number and do a call to the [PokémonTCG](https://pokemontcg.io/) api and pull the cards' stats if there's a match on the website.
   - The [PokémonTCG](https://pokemontcg.io/) website only has sets up to 2021 currently, so newer sets will not work with this example.
   - Make sure to take a relatively clear picture of the card. A good rule of thumb is, if you can't read the picture, neither can the program.
   - Special cards with unorthodox naming will have issues with this tool, the Mega Venasaur EX is about the limit of this tool.
   - The script has console logging so you know what its doing when its doing it:
    ![logging](https://github.com/user-attachments/assets/f41f77df-344f-47ee-a185-1f6fa9e0948d)

     







