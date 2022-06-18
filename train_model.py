from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="dataset/")
trainer.setTrainConfig(object_names_array=["without_mask", "mask_weared_incorrect", "with_mask"], batch_size=4, num_experiments=10, train_from_pretrained_model="pretrained-yolov3.h5")
trainer.trainModel()