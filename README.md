# SeasonVision

SeasonVision is a .NET computer vision library built on top of ONNX Runtime. It provides lightweight helpers for common inference scenarios such as image classification, object detection, face detection, face landmarks, face emotion recognition, face attribute analysis, and instance segmentation.

- Repository: [SeasonRealms/SeasonVision](https://github.com/SeasonRealms/SeasonVision)

- Models: https://huggingface.co/SeasonEngine/Vision

## Features

- Image classification with `MobileNet` and `Resnet`
- Face detection with `Ultraface`
- Face landmark detection with `FaceLandmark`
- Face emotion recognition with `FaceEmotion`
- Face attribute analysis with `FaceAttributes`
- Object detection with `FasterRcnn`
- Instance segmentation with `MaskRcnn`
- Shared image preprocessing utilities for RGBA image buffers

## Target Framework

- `net10.0`

## Installation

Add the package reference once a NuGet package is published:

```xml
<PackageReference Include="SeasonVision" Version="0.2.0" />
```

For local development, reference the project directly from your solution.

## Model Inputs

SeasonVision does not embed model assets. You are expected to provide compatible ONNX model files when calling the APIs.

The current source includes helpers for models similar to:

- UltraFace face detection
- FER+ emotion classification
- PIPNet face landmark detection
- OpenVINO age-gender-recognition-retail-0013
- Faster R-CNN object detection
- Mask R-CNN instance segmentation
- MobileNet v2 image classification
- ResNet50 image classification

Reference models used by the current helpers:

| Local File Name | Upstream Project | Official Weights Page | License | Notes |
|---|---|---|---|---|
| `Age-gender-recognition-retail-0013.onnx` | OpenVINO Open Model Zoo | [age-gender-recognition-retail-0013](https://docs.openvino.ai/2023.3/omz_models_model_age_gender_recognition_retail_0013.html) | `Apache-2.0` | Intel retail age/gender model from Open Model Zoo. |
| `Emotion-ferplus-8.onnx` | ONNX Model Zoo / FERPlus | [emotion-ferplus-8](https://huggingface.co/onnxmodelzoo/emotion-ferplus-8) | `MIT` | The model card body says `MIT`; the Hugging Face metadata tag also shows `apache-2.0`, so keep that mismatch in mind when redistributing. |
| `Mobilenetv2-12.onnx` | ONNX Model Zoo | [mobilenetv2-12](https://huggingface.co/onnxmodelzoo/mobilenetv2-12) | `Apache-2.0` | Image classification model used by `MobileNet`. |
| `Resnet50-v1-12-qdq.onnx` | ONNX Model Zoo | [resnet](https://github.com/onnx/models/tree/main/validated/vision/classification/resnet) | `Apache-2.0` | Quantized ResNet50 variant used by `Resnet`. |
| `Ultraface_version-RFB-320.onnx` | Ultra-Light-Fast-Generic-Face-Detector-1MB | [UltraFace](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) | `MIT` | `version-RFB-320` face detector export. |
| `Pipnet_r18_wflw_98.onnx` | PIPNet ONNX | [pipnet-onnx](https://github.com/yakhyo/pipnet-onnx/releases/tag/weights) | `MIT` | 98-point WFLW face landmark model. |
| `FasterRCNN-12-qdq.onnx` | ONNX Model Zoo | [faster-rcnn](https://github.com/onnx/models/tree/main/validated/vision/object_detection_segmentation/faster-rcnn) | `MIT` | Quantized Faster R-CNN R-50-FPN model. |
| `MaskRCNN-12-qdq.onnx` | ONNX Model Zoo | [mask-rcnn](https://github.com/onnx/models/tree/main/validated/vision/object_detection_segmentation/mask-rcnn) | `MIT` | Quantized Mask R-CNN R-50-FPN model. |

## API Overview

Most APIs accept:

- `model`: path to the ONNX model file
- `imageData`: RGBA byte buffer
- `width`: image width
- `height`: image height

Representative entry points:

- `Ultraface.Detect(...)`
- `FaceLandmark.Detect(...)`
- `FaceEmotion.Detect(...)`
- `FaceAttributes.Detect(...)`
- `FasterRcnn.Detect(...)`
- `MaskRcnn.Detect(...)`
- `MobileNet.Detect(...)`
- `Resnet.Detect(...)`

## Packaging

The project file is prepared for open-source distribution and NuGet packaging with:

- package metadata
- XML documentation generation
- repository metadata
- embedded package README
- symbol package generation with `snupkg`

To create a package locally:

```bash
dotnet pack SeasonVision/SeasonVision.csproj -c Release
```

## Notes

- The library currently uses `Microsoft.ML.OnnxRuntime.Managed`.
- Some models may require additional runtime/provider configuration depending on your deployment target.
- Annotated image outputs are returned as RGBA byte arrays.

## License

This project is released under the MIT License.
