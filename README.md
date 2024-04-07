# Sony Custom Layers (SCL)

Sony Custom Layers (SCL) is an open-source project implementing detection post process NN layers not supported by the TensorFlow Keras API or Torch's torch.nn for the easy integration of those layers into pretrained models.

## Table of Contents

- [Getting Started](#getting-started)
- [API](#API)
- [License](#license)


## Getting Started

This section provides an installation and a quick starting guide.

### Installation

To install the latest stable release of SCL, run the following command:
```
pip install sony-custom-layers
```
By default, no framework dependencies are installed.
To install SCL including the latest tested dependencies (up to patch version) for TensorFlow:
```
pip install sony-custom-layers[tf]
```
To install SCL including the latest tested dependencies (up to patch version) for PyTorch/ONNX/OnnxRuntime:
```
pip install sony-custom-layers[torch]
```
### Supported Versions

#### TensorFlow

| **Tested FW versions** | **Tested Python version** | **Serialization** |
|------------------------|---------------------------|-------------------|
| 2.10                   | 3.8-3.10                  | .h5               |
| 2.11                   | 3.8-3.10                  | .h5               |
| 2.12                   | 3.8-3.11                  | .h5  .keras       |
| 2.13                   | 3.8-3.11                  | .keras            |
| 2.14                   | 3.9-3.11                  | .keras            |
| 2.15                   | 3.9-3.11                  | .keras            |

#### PyTorch

| **Tested FW versions**                                                                                                   | **Tested Python version** | **Serialization**                                                               |
|--------------------------------------------------------------------------------------------------------------------------|---------------------------|---------------------------------------------------------------------------------|
| torch 2.0-2.2<br/>torchvision 0.15-0.17<br/>onnxruntime 1.15-1.17<br/>onnxruntime_extensions 0.8-0.10<br/>onnx 1.14-1.15 | 3.8-3.11                  | .onnx (via torch.onnx.export)<br/>.pt2 (via torch.export.export, torch2.2 only) |

## API
See https://sony.github.io/custom_layers

## Loading the model
### TensorFlow
```
with sony_custom_layers.keras.custom_layers_scope():
    model = tf.keras.models.load_model(path)
```
See [source](sony_custom_layers/keras/custom_objects.py) for further details.
### PyTorch
#### ONNX 
No special handling is required for torch.onnx.export and onnx.load  
To enable OnnxRuntime inference:
```
import onnxruntime as ort

from sony_custom_layers.pytorch import load_custom_ops

so = load_custom_ops(load_ort=True)
session = ort.InferenceSession(model_path, sess_options=so)
session.run(...)
```
Alternatively, you can pass your own SessionOptions object upon which to register the custom ops
```
load_custom_ops(ort_session_options=so)
```
#### PT2
To load a model exported by torch.export.export:
```
from sony_custom_layers.pytorch import load_custom_ops
load_custom_ops()
m = torch.export.load(model_path)
```
## License
[Apache License 2.0](LICENSE.md).


