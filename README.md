# Edge MDT Custom Layers (EdgeMDT CL) 

Edge MDT Custom Layers (EdgeMDT CL) is an open-source project implementing detection post process NN layers not supported by the TensorFlow Keras API or Torch's torch.nn for the easy integration of those layers into pretrained models.

## Table of Contents

- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Supported Versions](#supported-versions)
- [API](#api)
  - [TensorFlow API](#tensorflow-api)
  - [PyTorch API](#pytorch-api)
- [License](#license)


## Getting Started

This section provides an installation and a quick starting guide.

### Installation

To install the latest stable release of SCL, run the following command:
```
pip install edge-mdt-cl
```
By default, no framework dependencies are installed.
To install SCL including the latest tested dependencies (up to patch version) for TensorFlow:
```
pip install edge-mdt-cl[tf]
```
To install SCL including the latest tested dependencies (up to patch version) for PyTorch/ONNX/OnnxRuntime:
```
pip install edge-mdt-cl[torch]
```
### Supported Versions

#### TensorFlow

| **Tested FW versions** | **Tested Python version** | **Serialization** |
|------------------------|---------------------------|-------------------|
| 2.14                   | 3.9-3.11                  | .keras            |
| 2.15                   | 3.9-3.11                  | .keras            |

#### PyTorch

| **Tested FW versions**                                                                                                   | **Tested Python version** | **Serialization**              |
|--------------------------------------------------------------------------------------------------------------------------|---------------------------|--------------------------------|
| torch 2.3-2.6<br/>torchvision 0.18-0.21<br/>onnxruntime 1.15-1.21<br/>onnxruntime_extensions 0.8-0.13<br/>onnx 1.14-1.17 | 3.9-3.12                  | .onnx (via torch.onnx.export)  |

## API
For edge-mdt-cl API see https://sonysemiconductorsolutions.github.io/aitrios-edge-mdt-cl

### TensorFlow API
For TensorFlow layers see
[KerasAPI](https://sonysemiconductorsolutions.github.io/aitrios-edge-mdt-cl/edgemdt_cl/keras.html)

To load a model with custom layers in TensorFlow, see [custom_layers_scope](https://sonysemiconductorsolutions.github.io/aitrios-edge-mdt-cl/edgemdt_cl/keras.html#custom_layers_scope)

### PyTorch API
For PyTorch layers see
[PyTorchAPI](https://sonysemiconductorsolutions.github.io/aitrios-edge-mdt-cl/edgemdt_cl/pytorch.html)

No special handling is required for torch.onnx.export and onnx.load.

For OnnxRuntime support see [load_custom_ops](https://sonysemiconductorsolutions.github.io/aitrios-edge-mdt-cl/edgemdt_cl/pytorch.html#load_custom_ops) 

## License
[Apache License 2.0](LICENSE.md).


