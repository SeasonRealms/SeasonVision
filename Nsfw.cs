// Copyright (c) SeasonEngine and contributors.
// Licensed under the MIT License.
// https://github.com/SeasonRealms/SeasonVision

namespace Season.Vision;

/// <summary>
/// NSFW image classification inference for the GantMan nsfw_model family.
/// Matches the ONNX conversion's reference demos: resize directly to 299x299, scale
/// the pixels to 0..1 and feed NHWC. The exported graph already ends in softmax, so
/// the raw output is used as the class probabilities.
/// </summary>
public static class Nsfw
{
    private const int InputSize = 299;
    private const int TopK = 5;

    private static readonly string[] Categories = { "drawings", "hentai", "neutral", "porn", "sexy" };

    /// <summary>
    /// Indices of the categories that count as unsafe. The rest are benign by definition:
    /// drawings and neutral are safe for work in the source taxonomy.
    /// </summary>
    private static readonly int[] UnsafeClasses = { 1, 3, 4 };

    /// <summary>
    /// Classifies an RGBA image. Input pixels are expected to be RGBA8, matching the
    /// imageData convention of the other detectors in this assembly.
    /// </summary>
    public static NsfwResult Detect(InferenceSession session, ReadOnlySpan<byte> imageData, int width, int height)
    {
        // Resize directly to the input size, exactly as the conversion's reference demos do;
        // this is a plain stretch, not a shortest-edge resize followed by a center crop.
        var resized = ImageProcessor.Resize(imageData, width, height, InputSize, InputSize);

        var rgb = ImageProcessor.ExtractRgb(resized, InputSize, InputSize);

        var data = new float[InputSize * InputSize * 3];

        for (var i = 0; i < data.Length; i++)
        {
            data[i] = rgb[i] / 255f;
        }

        var tensor = new DenseTensor<float>(data, new[] { 1, InputSize, InputSize, 3 });

        var inputName = session.InputMetadata.Keys.FirstOrDefault() ?? "input";

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, tensor)
        };

        using var results = session.Run(inputs);

        var probabilities = results.First().AsEnumerable<float>().ToArray();

        return GetResult(probabilities, null);
    }

    /// <summary>
    /// Converts raw class probabilities into the public result, computing the unsafe score
    /// as the combined confidence of the nsfw categories.
    /// </summary>
    public static NsfwResult GetResult(float[] probabilities, float? threshold, int topK = TopK)
    {
        var result = new NsfwResult
        {
            Scores = probabilities,
            Threshold = threshold
        };

        if (probabilities is null || probabilities.Length == 0)
        {
            return result;
        }

        float unsafeScore = 0f;

        var predictions = new List<NsfwPrediction>(probabilities.Length);

        for (int i = 0; i < probabilities.Length; i++)
        {
            if (UnsafeClasses.Contains(i))
            {
                unsafeScore += probabilities[i];
            }

            predictions.Add(new NsfwPrediction
            {
                Label = i < Categories.Length ? Categories[i] : $"class_{i}",
                Confidence = probabilities[i]
            });
        }

        result.UnsafeScore = unsafeScore;

        var ordered = predictions.OrderByDescending(prediction => prediction.Confidence).ToList();

        result.TopLabel = ordered[0].Label;

        if (topK > 0 && ordered.Count > topK)
        {
            ordered = ordered.Take(topK).ToList();
        }

        result.Predictions = ordered;

        if (threshold.HasValue)
        {
            result.Blocked = unsafeScore >= threshold.Value;
        }

        return result;
    }

}

public sealed class NsfwPrediction
{
    public string Label { get; set; } = string.Empty;

    public float Confidence { get; set; }
}

public sealed class NsfwResult
{
    /// <summary>Full class probability vector, in model order.</summary>
    public float[] Scores { get; set; } = [];

    /// <summary>Sum of the unsafe categories probabilities.</summary>
    public float UnsafeScore { get; set; }

    /// <summary>Highest-probability category label.</summary>
    public string TopLabel { get; set; } = string.Empty;

    /// <summary>Top predictions ordered by confidence, for display.</summary>
    public List<NsfwPrediction> Predictions { get; set; } = new();

    /// <summary>True when <see cref="UnsafeScore"/> reached the threshold supplied by the caller.</summary>
    public bool Blocked { get; set; }

    /// <summary>The threshold used for the verdict, when one was supplied.</summary>
    public float? Threshold { get; set; }

    /// <summary>Human readable verdict and per-class confidence, one line per class.</summary>
    public string Summary
    {
        get
        {
            if (Predictions.Count == 0)
            {
                return string.Empty;
            }

            var builder = new StringBuilder();

            // The caller-supplied threshold is echoed so the measurement text is
            // self-contained: a score reads as a verdict only next to its cut-off.
            builder.AppendLine(Threshold.HasValue
                ? $"Verdict: {(Blocked ? "Blocked" : "Passed")}, UnsafeScore: {UnsafeScore:P2}, Threshold: {Threshold.Value:P2}, TopLabel: {TopLabel}"
                : $"Verdict: {(Blocked ? "Blocked" : "Passed")}, UnsafeScore: {UnsafeScore:P2}, TopLabel: {TopLabel}");

            foreach (var prediction in Predictions)
            {
                builder.AppendLine($"Label: {prediction.Label}, Confidence: {prediction.Confidence:P2}");
            }

            return builder.ToString().TrimEnd();
        }
    }
}
