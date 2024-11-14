import torch
import utils

def hook_fn(module, input, output):
    if isinstance(input, tuple):
        input = input[0]
    input_shape = tuple(input.shape) if input is not None else None
    output_shape = tuple(output.shape) if output is not None else None
    if "BatchNorm2d" in module.__class__.__name__:
        print(f"Module: {module}")
        print(f"Input shape: {input_shape}")
        print(f"Output shape: {output_shape}")
        print("-" * 50)

model = utils.load_model("swin_tiny_patch4_window7_224")
utils.replace_ln_with_bn(model)
# Register hook to all modules
for name, layer in model.named_modules():
    layer.register_forward_hook(hook_fn)

output = model(torch.randn(10, 3, 224, 224))
