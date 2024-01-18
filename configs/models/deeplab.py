import models.mapping


model_name = "DeepLabV3Plus"
dropout = 0.10
inference_dropout = False

model_builder, model_args = models.mapping.DeepLabV3Plus, {
    "last_activation": "softmax", 
    "dropout": dropout, 
    "inference_dropout": inference_dropout, 
    "name": model_name,
}

n_ds_branches = 1

weights_path = None
