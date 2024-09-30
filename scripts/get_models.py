import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



from models.helper import get_model 



model = get_model((224, 224), backbone_arch="vit_small_BitLinear", agg_arch="cls", normalize_output=True)


print(model)