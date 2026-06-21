# SeasonVision

SeasonVision is a .NET computer vision library built on top of ONNX Runtime. It provides lightweight helpers for common inference scenarios such as image classification, object detection, face detection, face landmarks, face emotion recognition, face attribute analysis, and instance segmentation.

- Repository: [SeasonRealms/SeasonVision](https://github.com/SeasonRealms/SeasonVision)

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
<PackageReference Include="SeasonVision" Version="0.1.0" />
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
