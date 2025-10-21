"""
Convert sklearn model to ONNX for browser deployment
"""

import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import os

# Load model
model = joblib.load('storage_model.pkl')
scaler = joblib.load('storage_scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

print(f"Model type: {type(model)}")
print(f"Features: {feature_names}")

# Define input type (5 features, float32)
initial_type = [('float_input', FloatTensorType([None, 5]))]

# Convert to ONNX
try:
    onnx_model = convert_sklearn(
        model, 
        initial_types=initial_type,
        target_opset=12
    )
    
    # Serialize to bytes
    onnx_bytes = onnx_model.SerializeToString()
    
    # Save ONNX model
    with open("storage_model.onnx", "wb") as f:
        f.write(onnx_bytes)
    
    # Get file size
    file_size = os.path.getsize("storage_model.onnx")
    
    print("\n✅ Model converted to ONNX successfully!")
    print("Saved: storage_model.onnx")
    print(f"File size: {file_size / 1024:.1f} KB")
    
    # Quick test with onnxruntime
    import onnxruntime as rt
    import numpy as np
    
    sess = rt.InferenceSession("storage_model.onnx")
    input_name = sess.get_inputs()[0].name
    
    # Test prediction
    test_input = np.array([[45.5, 25.0, 4, 500000.0, 50000.0]], dtype=np.float32)
    result = sess.run(None, {input_name: test_input})
    
    print(f"\n✅ Test prediction successful!")
    print(f"Sample input: {test_input[0]}")
    print(f"Predicted storage score: {float(result[0][0]):.2f}")
    
except Exception as e:
    print(f"❌ Conversion failed: {e}")
    import traceback
    traceback.print_exc()