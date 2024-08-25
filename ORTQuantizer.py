import os
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

# Define the quantization configuration
dqconfig = AutoQuantizationConfig.avx512_vnni(
    is_static=False,
    per_channel=False,
    use_symmetric_activations=True,
    operators_to_quantize=["MatMul"]
)

print(dqconfig)

# Define paths
og_onnx_dir = r'C:\Users\K_ADMIN\Desktop\AMD GenAI\smoothquant\LLama\onnx'
save_dir = r'C:\Users\K_ADMIN\Desktop\AMD GenAI\smoothquant\LLaMa_onnx_final'

# Load the quantizer with the specified ONNX model file
quantizer = ORTQuantizer.from_pretrained(og_onnx_dir, file_name="model.onnx")

# Perform quantization with external data format enabled
model_quantized_path = quantizer.quantize(
    save_dir=save_dir,
    quantization_config=dqconfig,
    use_external_data_format=True  # Enable external data format (cuz model too big)
)

print(f"Quantized model saved to: {model_quantized_path}")
