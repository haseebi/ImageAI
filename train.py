from imageai.Prediction.Custom import ModelTraining
import os

trainer = ModelTraining()
trainer.setModelTypeAsDenseNet()
trainer.setDataDirectory("Solan de Cabras")
trainer.trainModel(num_objects=1, num_experiments=50, enhance_data=True, batch_size=16, show_network_summary=True, save_full_model=True)
save_model_to_tensorflow(new_model_folder= os.path.join(execution_path, "tensorflow_model"), new_model_name="idenprof_resnet_tensorflow.pb")
