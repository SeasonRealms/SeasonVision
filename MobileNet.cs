// Copyright (c) SeasonEngine and contributors.
// Licensed under the MIT License.
// https://github.com/SeasonRealms/SeasonVision

namespace SeasonVision;

/// <summary>
/// MobileNet v2 image classification inference.
/// Matches the MauiVisionSample preprocessing flow: resize the shorter edge to
/// 256, then center-crop to 224x224.
/// </summary>
public static class MobileNet
{
    private const int ResizeShortestEdge = 256;
    private const int InputSize = 224;
    private const int TopK = 3;

    public static string Detect(string model, ReadOnlySpan<byte> imageData, int width, int height)
    {
        var rgb = ImageProcessor.ExtractRgb(imageData, width, height);
        var mean = new[] { 0.485f, 0.456f, 0.406f };
        var stddev = new[] { 0.229f, 0.224f, 0.225f };

        float[] chw = ImageProcessor.ResizeShortestEdgeCenterCropNormalize(
            rgb,
            width,
            height,
            ResizeShortestEdge,
            InputSize,
            InputSize,
            mean,
            stddev);

        var input = new DenseTensor<float>(chw, new[] { 1, 3, InputSize, InputSize });
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input", input)
        };

        using var session = new InferenceSession(model);
        using var results = session.Run(inputs);

        IEnumerable<float> output = results.First().AsEnumerable<float>();
        float sum = output.Sum(x => MathF.Exp(x));
        IEnumerable<float> softmax = output.Select(x => MathF.Exp(x) / sum);

        var topPredictions = softmax
            .Select((confidence, index) => new Prediction
            {
                Label = index < Resnet.Labels.Length ? Resnet.Labels[index] : $"class_{index}",
                Confidence = confidence
            })
            .OrderByDescending(x => x.Confidence)
            .Take(TopK)
            .ToList();

        var builder = new StringBuilder();
        builder.AppendLine($"Top {topPredictions.Count} predictions for MobileNet v2...");
        builder.AppendLine("--------------------------------------------------------------");

        foreach (var prediction in topPredictions)
        {
            builder.AppendLine($"Label: {prediction.Label}, Confidence: {prediction.Confidence:P2}");
        }

        return builder.ToString().TrimEnd();
    }

    private sealed class Prediction
    {
        public string Label { get; set; } = string.Empty;
        public float Confidence { get; set; }
    }
}
