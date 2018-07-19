Mask-RCNN-ActivityRecognition

To Run:

- Download Mask RCNN from https://github.com/matterport/Mask_RCNN
- FineTune your model, with your classes/objects segmentation. (A little dataset will be added soon)
- Use testVideo_createDataset.py to create your own Dataset for the LSTM. An example is in the folder LSTM/Dataset_LSTM
- Launch the LSTM,changing the batch, sequence lenght etc. with 
	Train: python lstm_activityRecognition.py --run_opt=1 --batch_size=256 --num_epochs=460 --sequence_length=20
	Test: python lstm_activityRecognition.py --run_opt=2 --batch_size=256 --num_epochs=460 --sequence_length=20

Remember to change in the code some paths, and to insert your classes and objects name correctly. 


