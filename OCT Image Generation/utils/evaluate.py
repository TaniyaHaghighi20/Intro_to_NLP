from copy import deepcopy
import lightning.pytorch as pl
import matplotlib.pyplot as plt

from utils.log_results import log_results


def evaluate_model(model, classes, dataset_name, data_modules, devices, comments, epochs=100):
    for data_module in data_modules:
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=devices,
            max_epochs=epochs,
        )
        test_loader = data_module.filtered_test_dataloader(classes)
        model.test_classes = data_module.filtered_classes
        if len(model.test_classes) == 0:  #it is not possible to test it!
            continue
        test_results = trainer.test(model, dataloaders=test_loader, verbose=False)
        # Ensure test_preds and test_labels are on CPU and converted to numpy arrays
        # ###
        log_results(classes=data_module.filtered_classes,
                    results=test_results,
                    comments=dataset_name + "_" + comments,
                    test_name=data_module.dataset_name,
                    approach="centralized",
                    epochs=epochs)
