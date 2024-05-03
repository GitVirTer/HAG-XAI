# HAG-XAI

Human Attention Guided eXplainable AI (HAG-XAI) can leverage human attention information to guide the combination of explanatory information, including activation maps and gradient maps, for gradient-based XAI methods in computer vision models, which can potentially enhance their plausibility and faithfulness.
![alt text](http://guoyang.work/data/Graphical_Abstract_1.png?raw=true)

## Citation

Please cite the tool using this website repository and the manuscript:

- Liu, G., Zhang, J., Chan, A. B., & Hsiao, J. (2023). Human attention-guided explainable AI for object detection. Proceedings of the Annual Meeting of the Cognitive Science Society.
- Liu, G., Zhang, J., Chan, A. B., & Hsiao, J. (2023). Human Attention-Guided Explainable Artificial Intelligence for Computer Vision Models. Neural Networks (Under Review).

## Usage

This repository consists of four folders.

- 'HAG-XAI_for_ObjectDetectionModels': The HAG-XAI training program for object detection model (Yolo-v5s).
  - Step 1: Run 'Main_Format_Database_Yolov5s.m' to generate the training database.
  - Step 2: Run 'Main_Train_HAGXAI_Yolov5s.m' to train the HAG-XAI model.
 
  Running 'Main_Test_One_Sample_Yolov5s.m' can generate the HAG-XAI saliency map for one sample.
  The experimental materials and human attention data can be downloaded here: http://guoyang.work/data/Data.zip
  Note that Matlab 2021a or a later version with the deep learning toolbox is required for running Matlab codes.
  
- 'HAG-XAI_for_ImageClassificationModels': The HAG-XAI training program for image classification models.
  - 'Main_HAGXAI_Train.m' is the training program.
  - 'Image_Classification_Stimuli_ImageNet.zip' is the zipped experimental database.
  
  The experimental training data can be downloaded here: http://guoyang.work/data/TrainingDatabase_GradAct_resnet50_Cls.zip

  The model files for pretrained ResNet-50 and Xception can be downloaded here:
  http://guoyang.work/data/ImgClsModelFiles_Xception_ResNet50.zip
  
- 'FullGradCAM_Yolov5s': The FullGradCAM algorithm for Yolo-v5s model. Main function: 'main.py'. 'yolov5sbdd100k300epoch.pt' is the Yolo-v5s model file trained on BDD-100K. 'yolov5s_COCOPretrained.pt' is the Yolo-v5s model file pretrained on MS-COCO.

- 'FullGradCAM_FasterRCNN': The FullGradCAM algorithm for Faster-RCNN model. Main function: 'main.py'. The model file can be downloaded here: http://guoyang.work/data/FasterRCNN_C4_BDD100K.zip

## Contact

If there is any technical issue, please contact the author Guoyang Liu: gyliu@sdu.edu.cn (please also cc virter1995@outlook.com) 
Visit http://guoyang.work/ for more related works.


