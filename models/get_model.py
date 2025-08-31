import sys
import os
models_dir = os.path.dirname(os.path.abspath(__file__))
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)
import timm  
import torch
import torchvision.models as models
import torch.nn as nn
from mobilevit import mobilevit_s, mobilevit_xxs, mobilevit_xs
from mobilevitv2 import mobilevitv2_xxs, mobilevitv2_xs, mobilevitv2_s
from mobilevitv3 import mobilevitv3_xxs, mobilevitv3_xs, mobilevitv3_s
from wmvit import wmvit_s, wmvit_xxs, wmvit_xs, wmvit_050, wmvit_060, wmvit_075, wmvit_080

def get_model(model_name: str, num_classes: int, INPUT_SIZE: int, **kwargs):
    """
    Model factory function: Returns a model instance based on the model_name.
    - First, check if it is your custom model.
    - Then, attempt to load the model from the timm library.
    - timm will automatically handle the replacement of the classification head, 
      and we only need to pass in num_classes.
    """
    model_name = model_name.lower()
    model = None

    # ==================== Our Model  ====================
    if model_name.startswith("wmvit"):
        ratio = kwargs.get('ratio', 1.0)
        if "xxs" in model_name:
            model = wmvit_xxs(num_classes=num_classes, ratio=ratio)
        elif "xs" in model_name:
            model = wmvit_xs(num_classes=num_classes, ratio=ratio)
        elif "s" in model_name:
            model = wmvit_s(num_classes=num_classes, ratio=ratio)
        elif "050" in model_name:
            model = wmvit_050(num_classes=num_classes, ratio=ratio)
        elif "060" in model_name:
            model = wmvit_060(num_classes=num_classes, ratio=ratio)
        elif "075" in model_name:
            model = wmvit_075(num_classes=num_classes, ratio=ratio) 
        elif "080" in model_name:
            model = wmvit_080(num_classes=num_classes, ratio=ratio)


    elif model_name.startswith("mobilevitv3"):
        ratio = kwargs.get('ratio', 1.0)
        if "xxs" in model_name:
            model = mobilevitv3_xxs(num_classes=num_classes, ratio=ratio)
        elif "xs" in model_name:
            model = mobilevitv3_xs(num_classes=num_classes, ratio=ratio)
        elif "s" in model_name:
            model = mobilevitv3_s(num_classes=num_classes, ratio=ratio)

    elif model_name == "mobilevit_xxs":
        model = mobilevit_xxs(num_classes=num_classes)
    elif model_name == "mobilevit_xs":
        model = mobilevit_xs(num_classes=num_classes)
    elif model_name == "mobilevit_s":
        model = mobilevit_s(num_classes=num_classes)

    elif model_name == "mobilevitv2_xxs":
        model = mobilevitv2_xxs(num_classes=num_classes)
    elif model_name == "mobilevitv2_xs":
        model = mobilevitv2_xs(num_classes=num_classes)
    elif model_name == "mobilevitv2_s":
        model = mobilevitv2_s(num_classes=num_classes)
        
    elif model_name == 'shufflenet_v2_x1_0':
        model = models.shufflenet_v2_x1_0(weights=None)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
        
    # elif model_name.startswith("mobilemamba"):
    #     if model_name == "mobilemamba_t2":
    #         model = MobileMamba_S6(num_classes=num_classes)
    #     elif model_name == "mobilemamba_s6":
    #         model = MobileMamba_B1(num_classes=num_classes)
    #     elif model_name == "mobilemamba_b1":
    #         model = MobileMamba_B1(num_classes=num_classes)
    #     elif model_name == "mobilemamba_b2":
    #         model = MobileMamba_B4(num_classes=num_classes)
    #     else:
    #         raise ValueError(f"Unknown MobileMamba variant: {model_name}")
        
    # ==================== From timm Library  ====================
    else:
        try:
            # Parameter description of the timm.create_model() function:
            # - `num_classes`: Automatically replace the classification head 
            #    with the default value being false
            model = timm.create_model(model_name, num_classes=num_classes)
            
        except Exception as e:
            print(f"Info: Model '{model_name}' not found in timm library. Error: {e}")
            model = None

    if model is None:
        raise ValueError(f"Model '{model_name}' is not recognized in custom models or the timm library. Please check the spelling.")
    return model