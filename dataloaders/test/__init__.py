from .Tokyo247Dataset import Tokyo247
from .SVOXDataset import SVOX_Night, SVOX_Sun, SVOX_Rain, SVOX_Overcast, SVOX_Snow
from .PittsburghDataset import Pitts30k, Pitts250k
from .MapillaryDataset import MSLS
from .AmsterTimeDataset import AmsterTime
from .EynshamDataset import Eynsham

__all__ = ["Tokyo247", "SVOX_Night", "SVOX_Sun", "SVOX_Rain", "SVOX_Overcast", "SVOX_Snow", "Pitts30k", "Pitts250k", "MSLS", "AmsterTime", "Eynsham"]


TEST_DATASETS = {
    "tokyo247": Tokyo247,
    "svoxt_night": SVOX_Night,
    "svoxt_sun": SVOX_Sun,
    "svoxt_rain": SVOX_Rain,
    "svoxt_overcast": SVOX_Overcast,
    "svoxt_snow": SVOX_Snow,
    "pitts30k": Pitts30k,
    "pitts250k": Pitts250k,
    "msls": MSLS,
    "amster_time": AmsterTime,
    "eynsham": Eynsham,
}