from .AmsterTimeDataset import AmsterTime
from .EynshamDataset import Eynsham
from .MapillaryDataset import MSLS
from .PittsburghDataset import Pitts30k, Pitts250k
from .sfxl import SFXLOcclusion
from .stlucia import StLucia
from .SVOXDataset import (SVOX_Night, SVOX_Overcast, SVOX_Rain, SVOX_Snow,
                          SVOX_Sun)
from .Tokyo247Dataset import Tokyo247

__all__ = [
    "Tokyo247",
    "SVOX_Night",
    "SVOX_Sun",
    "SVOX_Rain",
    "SVOX_Overcast",
    "SVOX_Snow",
    "Pitts30k",
    "Pitts250k",
    "MSLS",
    "AmsterTime",
    "Eynsham",
    "SFXLOcclusion",
    "StLucia",
]


TEST_DATASETS = {
    "tokyo247": Tokyo247,
    "svox_night": SVOX_Night,
    "svox_sun": SVOX_Sun,
    "svox_rain": SVOX_Rain,
    "svox_overcast": SVOX_Overcast,
    "svox_snow": SVOX_Snow,
    "pitts30k": Pitts30k,
    "pitts250k": Pitts250k,
    "msls": MSLS,
    "amster_time": AmsterTime,
    "eynsham": Eynsham,
    "sfxlocclusion": SFXLOcclusion,
    "stlucia": StLucia,
}
