"""
datamodule.py: Handles the initialization of data modules
__author: Sina Gholami
__update: add comments
__update_date: 7/5/2024
"""
import os
import numpy as np
from dotenv import load_dotenv

from dataset.datamodule import SrinivasanDataModule, NurDataModule


def get_datamodule(dataset_name,
                   dataset_path,
                   batch_size,
                   train_transform,
                   test_transform,
                   srinivasan_classes=None,
                   nur_classes=None,
                   ):
    """
    auxiliary function to create and return a datamodule based on the dataset name
    """
    datamodule = None
    # if dataset_name == "DS1":
    #     datamodule = KermanyDataModule(dataset_name=dataset_name,
    #                                    data_dir=dataset_path,
    #                                    batch_size=batch_size,
    #                                    classes=kermany_classes,
    #                                    split=[0.9, 0.025, 0.075],
    #                                    train_transform=train_transform,
    #                                    test_transform=test_transform,
    #                                    )
    #     # preparing config
    #     datamodule.prepare_data()
    #     datamodule.setup("train")
    #     datamodule.setup("val")
    #     datamodule.setup("test")

    if dataset_name == "DS2":
        datamodule = SrinivasanDataModule(dataset_name=dataset_name,
                                          data_dir=dataset_path,
                                          batch_size=batch_size,
                                          classes=srinivasan_classes,
                                          split=[0.67, 0.13, 0.33],
                                          train_transform=train_transform,
                                          test_transform=test_transform,
                                          )
        # preparing config
        datamodule.prepare_data()
        datamodule.setup("train")
        datamodule.setup("val")
        datamodule.setup("test")

    # elif dataset_name == "DS3":
    #     datamodule = OCT500DataModule(dataset_name=dataset_name,
    #                                   data_dir=dataset_path,
    #                                   batch_size=batch_size,
    #                                   classes=oct500_classes,
    #                                   split=[0.85, 0.05, 0.1],
    #                                   train_transform=train_transform,
    #                                   test_transform=test_transform,
    #                                   filter_img=OCT500_filter_img,
    #                                   merge=OCT500_merge,
    #                                   threemm=OCT500_threemm
    #                                   )
    #     # preparing config
    #     datamodule.prepare_data()
    #     datamodule.setup("train")
    #     datamodule.setup("val")
    #     datamodule.setup("test")

    elif dataset_name == "DS4":
        datamodule = NurDataModule(dataset_name=dataset_name,
                                   data_dir=dataset_path,
                                   batch_size=batch_size,
                                   classes=nur_classes,
                                   split=[0.8, 0.05, 0.15],
                                   train_transform=train_transform,
                                   test_transform=test_transform,
                                   )
        # preparing config
        datamodule.prepare_data()
        datamodule.setup("train")
        datamodule.setup("val")
        datamodule.setup("test")

    # elif dataset_name == "DS5":
    #     datamodule = WaterlooDataModule(dataset_name=dataset_name,
    #                                     data_dir=dataset_path,
    #                                     batch_size=batch_size,
    #                                     classes=waterloo_classes,
    #                                     split=[0.7, 0.1, 0.2],
    #                                     train_transform=train_transform,
    #                                     test_transform=test_transform,
    #                                     )
    #     # preparing config
    #     datamodule.prepare_data()
    #     datamodule.setup("train")
    #     datamodule.setup("val")
    #     datamodule.setup("test")

    # elif dataset_name == "DS6":
    #     datamodule = OCTDLDataModule(dataset_name=dataset_name,
    #                                  data_dir=dataset_path,
    #                                  batch_size=batch_size,
    #                                  classes=octdl_classes,
    #                                  split=[0.7, 0.10, 0.2],
    #                                  train_transform=train_transform,
    #                                  test_transform=test_transform,
    #                                  )
    #     # preparing config
    #     datamodule.prepare_data()
    #     datamodule.setup("train")
    #     datamodule.setup("val")
    #     datamodule.setup("test")


    # elif dataset_name == "DS10":
    #     datamodule = OIMHSDataModule(dataset_name=dataset_name,
    #                                  data_dir=dataset_path,
    #                                  batch_size=batch_size,
    #                                  train_transform=train_transform,
    #                                  val_transform=test_transform,
    #                                  classes=oimhs_classes,
    #                                  )
    #     # preparing config
    #     datamodule.prepare_data()
    #     datamodule.setup("None")

    # elif dataset_name == "DS11":
    #     datamodule = OLIVEDataModule(dataset_name=dataset_name,
    #                                  data_dir=dataset_path,
    #                                  batch_size=batch_size,
    #                                  train_transform=train_transform,
    #                                  val_transform=test_transform,
    #                                  classes=olive_classes,
    #                                  )
    #     datamodule.prepare_data()
    #     datamodule.setup("None")

    # elif dataset_name == "DS13":
    #     datamodule = FairhubDataModule(dataset_name=dataset_name,
    #                                  data_dir=dataset_path,
    #                                  batch_size=batch_size,
    #                                  train_transform=train_transform,
    #                                  val_transform=test_transform,
    #                                  split=[1],
    #                                  classes=fairhub_classes,
    #                                  )
                                     
    #     # preparing config
    #     datamodule.prepare_data()
    #     datamodule.setup("None")
    

    return datamodule


def get_data_modules(batch_size, classes, train_transform=None, test_transform=None, 
                     env_path="./data/.env", pretrain=True):
    """
    create and return all available data modules
    returns a list of data modules
    """
    load_dotenv(dotenv_path=env_path)  #read the dataset paths from the .env file
    # client_name = "DS1"
    print("loading")
    # DATASET_PATH = os.getenv(client_name + "_PATH")
    # kermany_datamodule = get_datamodule(dataset_name=client_name,
    #                                     dataset_path=DATASET_PATH,
    #                                     batch_size=batch_size,
    #                                     kermany_classes=classes[0],
    #                                     train_transform=train_transform,
    #                                     test_transform=test_transform
    #                                     )
    # print("loading kermany completed")

    client_name = "DS2"
    DATASET_PATH = os.getenv(client_name + "_PATH")
    srinivasan_datamodule = get_datamodule(dataset_name=client_name,
                                           dataset_path=DATASET_PATH,
                                           batch_size=batch_size,
                                           srinivasan_classes=classes[1],
                                           train_transform=train_transform,
                                           test_transform=test_transform
                                           )

    # client_name = "DS3"
    # DATASET_PATH = os.getenv(client_name + "_PATH")
    # oct500_datamodule = get_datamodule(dataset_name=client_name,
    #                                    dataset_path=DATASET_PATH,
    #                                    batch_size=batch_size,
    #                                    oct500_classes=classes[2],
    #                                    train_transform=train_transform,
    #                                    test_transform=test_transform,
    #                                    OCT500_filter_img=OCT500_filter_img,
    #                                    OCT500_merge=OCT500_merge,
    #                                    OCT500_threemm=OCT500_threemm
    #                                    )

    client_name = "DS4"
    DATASET_PATH = os.getenv(client_name + "_PATH")
    nur_datamodule = get_datamodule(dataset_name=client_name,
                                    dataset_path=DATASET_PATH,
                                    batch_size=batch_size,
                                    nur_classes=classes[3],
                                    train_transform=train_transform,
                                    test_transform=test_transform,
                                    )
    # client_name = "DS5"
    # DATASET_PATH = os.getenv(client_name + "_PATH")
    # waterloo_datamodule = get_datamodule(dataset_name=client_name,
    #                                      dataset_path=DATASET_PATH,
    #                                      batch_size=batch_size,
    #                                      waterloo_classes=classes[4],
    #                                      train_transform=train_transform,
    #                                      test_transform=test_transform,
    #                                      )

    # client_name = "DS6"
    # DATASET_PATH = os.getenv(client_name + "_PATH")
    # octdl_datamodule = get_datamodule(dataset_name=client_name,
    #                                   dataset_path=DATASET_PATH,
    #                                   batch_size=batch_size,
    #                                   octdl_classes=classes[5],
    #                                   train_transform=train_transform,
    #                                   test_transform=test_transform,
    #                                   )

    # client_name = "DS8"
    # DATASET_PATH = os.getenv(client_name + "_PATH")
    # mario_datamodule = get_datamodule(dataset_name=client_name,
    #                                 dataset_path=DATASET_PATH,
    #                                 batch_size=batch_size,
    #                                 mario_classes=classes[7],
    #                                 train_transform=train_transform,
    #                                 test_transform=test_transform,
    #                                 mario_set_label=mario_set_label
    #                                 )
    data_modules = [
        # kermany_datamodule,
        srinivasan_datamodule,
        # oct500_datamodule,
        nur_datamodule,
        # waterloo_datamodule,
        # octdl_datamodule,
        # uic_dr_datamodule,
        # mario_datamodule,
    ]


    #     client_name = "DS10"
    #     DATASET_PATH = os.getenv(client_name + "_PATH")
    #     oimhs_datamodule = get_datamodule(dataset_name=client_name,
    #                                     dataset_path=DATASET_PATH,
    #                                     batch_size=batch_size,
    #                                     oimhs_classes=classes[9],
    #                                     train_transform=train_transform,
    #                                     test_transform=test_transform,
    #                                     )
                                        
    
        
    #     client_name = "DS11"
    #     DATASET_PATH = os.getenv(client_name + "_PATH")
    #     olive_datamodule = get_datamodule(dataset_name=client_name,
    #                                     dataset_path=DATASET_PATH,
    #                                     batch_size=batch_size,
    #                                     olive_classes=classes[10],
    #                                     train_transform=train_transform,
    #                                     test_transform=test_transform,
    #                                     )

    #     client_name = "DS13"
    #     DATASET_PATH = os.getenv(client_name + "_PATH")
    #     fairhub_datamodule = get_datamodule(dataset_name=client_name,
    #                                     dataset_path=DATASET_PATH,
    #                                     batch_size=batch_size,
    #                                     fairhub_classes=classes[12],
    #                                     train_transform=train_transform,
    #                                     test_transform=test_transform,
    #                                     )
    #     data_modules.append(wf_datamodule)
    #     data_modules.append(oimhs_datamodule)
    #     data_modules.append(olive_datamodule)
    #     data_modules.append(thoct_datamodule)
    #     data_modules.append(fairhub_datamodule)

    return data_modules

