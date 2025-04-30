def get_merged_classes():
    kermany_classes = {"NORMAL": 0,
                       "DRUSEN": 1,
                       "DME": 2,
                       "CNV": 3,
                       }

    srinivasan_classes = {"NORMAL": 0,
                          "AMD": 1,
                          "DME": 2,
                          }

    oct500_classes = {"NORMAL": 0,
                      "AMD": 1,
                      "DR": 2,
                      "OTHERS": 3,
                      }

    nur_classes = {"NORMAL": 0,
                   "DRUSEN": 1,
                   "CNV": 2,
                   }

    waterloo_classes = {"NORMAL": 0,
                        "AMD": 1,
                        "DR": 2,
                        "HR": 3,
                        "CSR": 4
                        }

    octdl_classes = {"NORMAL": 0,
                     "AMD": 1,
                     "DME": 2,
                     "ERM": 3,
                     "RAO": 4,
                     "RVO": 5,
                     "VID": 6
                     }
    uic_dr_classes = {"Control": 0,
                      "Mild": 1,
                      "Moderate": 2,
                      "Severe": 3,
                      }
    mario_classes = {"Reduced": 0,
                     "Stable": 1,
                     "Increased": 2,
                     "uninterpretable": 3,
                     "None": -1
                     }
    wf_classes = {}
    oimhs_classes = {"1": 0,
                     "2": 1,
                     "3": 2,
                     "4": 3,
                     }
    olive_classes = {}
    thoct_classes = {"NORMAL": 0,
                    "AMD": 1,
                    "DME": 2,
                    }
    fairhub_classes = {}
    return (kermany_classes, srinivasan_classes, oct500_classes, nur_classes, waterloo_classes, octdl_classes,
            uic_dr_classes, mario_classes, wf_classes, oimhs_classes, olive_classes, thoct_classes, fairhub_classes)


def get_full_classes():
    kermany_classes = {"NORMAL": 0,
                       "DRUSEN": 1,
                       "DME": 2,
                       "CNV": 3,
                       }

    srinivasan_classes = {"NORMAL": 0,
                          "AMD": 1,
                          "DME": 2,
                          }

    oct500_classes = {"NORMAL": 0,
                      "AMD": 1,
                      "DR": 2,
                      "CNV": 3,
                      "OTHERS": 4,
                      "RVO": 5,
                      "CSC": 6,
                      }

    nur_classes = {"NORMAL": 0,
                   "DRUSEN": 1,
                   "CNV": 2,
                   }

    waterloo_classes = {"NORMAL": 0,
                        "AMD": 1,
                        "DR": 2,
                        "HR": 3,
                        "CSR": 4
                        }

    octdl_classes = {"NORMAL": 0,
                     "AMD": 1,
                     "DME": 2,
                     "ERM": 3,
                     "RAO": 4,
                     "RVO": 5,
                     "VID": 6
                     }

    oimhs_classes = {"1": 0,
                     "2": 1,
                     "3": 2,
                     "4": 3,
                     }
    olive_classes = {}

    fairhub_classes = {}
    return (kermany_classes, srinivasan_classes, oct500_classes, nur_classes, waterloo_classes, octdl_classes,
          oimhs_classes, olive_classes, fairhub_classes)


def get_diffusion_classes(): #19 different categories
    kermany_classes = {"NORMAL": 0,
                       "DRUSEN": 1,
                       "DME": 2,
                       "CNV": 3,
                       }

    srinivasan_classes = {"NORMAL": 0,
                          "AMD": 4,
                          "DME": 2,
                          }

    oct500_classes = {"NORMAL": 0,
                      "AMD": 4,
                      "DR": 5,
                      "CNV": 3,
                      "OTHERS": 6,
                      "RVO": 7,
                      "CSC": 8,
                      }

    nur_classes = {"NORMAL": 0,
                   "DRUSEN": 1,
                   "CNV": 3,
                   }

    waterloo_classes = {"NORMAL": 0,
                        "AMD": 4,
                        "DR": 5,
                        "HR": 9,
                        "CSR": 10
                        }

    octdl_classes = {"NORMAL": 0,
                     "AMD": 4,
                     "DME": 2,
                     "ERM": 11,
                     "RAO": 12,
                     "RVO": 7,
                     "VID": 13
                     }
    oimhs_classes = {"1": 0,
                     "2": 14,
                     "3": 15,
                     "4": 16,
                     }
    olive_classes = {"Random":20}
    fairhub_classes = {"Random":20}
    return (kermany_classes, srinivasan_classes, oct500_classes, nur_classes, waterloo_classes, octdl_classes,
             oimhs_classes, olive_classes, fairhub_classes)
