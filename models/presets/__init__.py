from .dinoboq import DinoBoQ
from .dinosalad import DinoSalad
from .resnetboq import ResNetBoQ
from .cosplace import CosPlaceR18D128, CosPlaceR18D32, CosPlaceR18D64, CosPlaceR50D128, CosPlaceR50D2048, CosPlaceR50D32, CosPlaceR50D64
from .eigenplaces import EigenPlacesR18D256, EigenPlacesR18D512, EigenPlacesR50D128, EigenPlacesR50D2048, EigenPlacesR50D256, EigenPlacesR50D512


ALL_BASELINES = {
    "dinobqo": DinoBoQ,
    "dinosalad": DinoSalad,
    "resnetboq": ResNetBoQ,
    "cosplace": CosPlaceR50D64,
    "cosplacer18d128": CosPlaceR18D128,
    "cosplacer18d32": CosPlaceR18D32,
    "cosplacer18d64": CosPlaceR18D64,
    "cosplacer50d128": CosPlaceR50D128,
    "cosplacer50d2048": CosPlaceR50D2048,
    "cosplacer50d32": CosPlaceR50D32,
    "cosplacer50d64": CosPlaceR50D64,
    "eigenplacesr18d256": EigenPlacesR18D256,
    "eigenplacesr18d512": EigenPlacesR18D512,
    "eigenplacesr50d128": EigenPlacesR50D128,
    "eigenplacesr50d2048": EigenPlacesR50D2048,
    "eigenplacesr50d256": EigenPlacesR50D256,
    "eigenplacesr50d512": EigenPlacesR50D512,
}