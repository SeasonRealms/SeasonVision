// Copyright (c) SeasonEngine and contributors.
// Licensed under the MIT License.
// https://github.com/SeasonRealms/SeasonVision

namespace Season.Vision;

/// <summary>
/// Face emotion recognition.
/// The current implementation follows the FER+ ONNX input and output contract:
/// - Input: 1x1x64x64 grayscale image
/// - Output: 1x8 emotion logits converted to probabilities with softmax
/// </summary>
public static class FaceEmotion
{
    private static readonly string[] DefaultLabels =
    [
        "Neutral",
        "Happiness",
        "Surprise",
        "Sadness",
        "Anger",
        "Disgust",
        "Fear",
        "Contempt"
    ];

    private const float CropPadding = 0.12f;
    private const int FontSize = 16;
    private const int MaxRankedScores = 3;

    public static FaceEmotionResult Detect(
        InferenceSession detectorSession,
        InferenceSession recognizerSession,
        ReadOnlySpan<byte> imageData, int width, int height,
        bool createAnnotatedImage = false,
        int maxFaces = 5)
    {
        if (maxFaces <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxFaces), "maxFaces must be greater than 0.");

        var config = GetModelConfig(recognizerSession);
        var detectorFaces = Ultraface.Detect(detectorSession, imageData, width, height, false, maxFaces);
        var rgb = ImageProcessor.ExtractRgb(imageData, width, height);

        var result = new FaceEmotionResult
        {
            Model = null, //model,
            DetectorModel = null, // detectorModel,
            ImageWidth = width,
            ImageHeight = height,
            RequestedMaxFaces = maxFaces,
            InputWidth = config.InputWidth,
            InputHeight = config.InputHeight,
            EmotionLabels = config.Labels.ToArray()
        };

        foreach (var detectorFace in detectorFaces.Faces)
        {
            var crop = ExpandSquareCrop(detectorFace.Box, width, height);
            var cropRgb = ImageProcessor.CropRgb(rgb, width, height, crop.X, crop.Y, crop.Size, crop.Size);
            var chw = ResizeToGrayNchw(cropRgb, crop.Size, crop.Size, config.InputWidth, config.InputHeight);

            var input = new DenseTensor<float>(chw, new[] { 1, 1, config.InputHeight, config.InputWidth });
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(config.InputName, input)
            };

            using var outputs = recognizerSession.Run(inputs);
            var prediction = Decode(outputs, config.Labels);

            result.Faces.Add(new FaceEmotionFace
            {
                Index = result.Faces.Count,
                DetectionConfidence = detectorFace.Confidence,
                BoundingBox = new FaceLandmarkBox(
                    detectorFace.Box.Xmin,
                    detectorFace.Box.Ymin,
                    detectorFace.Box.Xmax,
                    detectorFace.Box.Ymax),
                CropBox = new FaceLandmarkBox(
                    crop.X,
                    crop.Y,
                    crop.X + crop.Size - 1,
                    crop.Y + crop.Size - 1),
                Emotion = prediction.Emotion,
                EmotionConfidence = prediction.EmotionConfidence,
                Scores = prediction.Scores
            });
        }

        if (createAnnotatedImage)
            result.AnnotatedImage = DrawResults(imageData, width, height, result.Faces);

        return result;
    }

    private static DecodedEmotion Decode(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs,
        IReadOnlyList<string> labels)
    {
        var tensors = outputs.ToArray();
        if (tensors.Length == 0)
            throw new InvalidOperationException("FaceEmotion produced no outputs.");

        Tensor<float>? scoreTensor = null;
        foreach (var tensor in tensors)
        {
            var candidate = tensor.AsTensor<float>();
            if (candidate.Length == labels.Count)
            {
                scoreTensor = candidate;
                break;
            }
        }

        if (scoreTensor == null)
            scoreTensor = tensors[0].AsTensor<float>();

        var logits = scoreTensor.ToArray();
        if (logits.Length != labels.Count)
            throw new NotSupportedException($"Only models with {labels.Count} emotion outputs are supported. Actual output length: {logits.Length}.");

        var probabilities = Softmax(logits);
        int bestIndex = 0;
        for (int i = 1; i < probabilities.Length; i++)
        {
            if (probabilities[i] > probabilities[bestIndex])
                bestIndex = i;
        }

        var scores = probabilities
            .Select((score, index) => new FaceEmotionScore
            {
                Label = labels[index],
                Score = score
            })
            .OrderByDescending(item => item.Score)
            .Take(MaxRankedScores)
            .ToList();

        return new DecodedEmotion(
            Emotion: labels[bestIndex],
            EmotionConfidence: probabilities[bestIndex],
            Scores: scores);
    }

    private static byte[] DrawResults(ReadOnlySpan<byte> imageData, int width, int height, List<FaceEmotionFace> faces)
    {
        if (faces.Count == 0)
            return imageData.ToArray();

        var pixels = ImageProcessor.EnsureRgba(imageData, width, height);
        int thickness = Math.Max(2, Math.Min(width, height) / 300);

        foreach (var face in faces)
        {
            DrawRectOutline(pixels, width, height, face.BoundingBox, thickness, 255, 0, 0);
            //string text = $"{face.Emotion} {face.EmotionConfidence:P0}";
            //DrawLabelText(
            //    pixels,
            //    width,
            //    height,
            //    (int)face.BoundingBox.Xmin,
            //    (int)face.BoundingBox.Ymin,
            //    text,
            //    255,
            //    255,
            //    255);
        }

        return pixels;
    }

    private static FaceSquareCrop ExpandSquareCrop(UltrafaceBox box, int imageWidth, int imageHeight)
    {
        float faceWidth = box.Xmax - box.Xmin + 1f;
        float faceHeight = box.Ymax - box.Ymin + 1f;
        float side = Math.Max(faceWidth, faceHeight) * (1f + CropPadding * 2f);

        float centerX = (box.Xmin + box.Xmax) * 0.5f;
        float centerY = (box.Ymin + box.Ymax) * 0.5f;

        int size = Math.Max(1, (int)MathF.Round(side));
        int x = (int)MathF.Round(centerX - size * 0.5f);
        int y = (int)MathF.Round(centerY - size * 0.5f);

        x = Math.Clamp(x, 0, Math.Max(0, imageWidth - size));
        y = Math.Clamp(y, 0, Math.Max(0, imageHeight - size));

        if (x + size > imageWidth)
            size = imageWidth - x;

        if (y + size > imageHeight)
            size = Math.Min(size, imageHeight - y);

        size = Math.Max(1, size);
        return new FaceSquareCrop(x, y, size);
    }

    private static float[] ResizeToGrayNchw(byte[] srcRgb, int srcW, int srcH, int dstW, int dstH)
    {
        float scaleX = (float)srcW / dstW;
        float scaleY = (float)srcH / dstH;
        float[] result = new float[dstW * dstH];

        for (int dy = 0; dy < dstH; dy++)
        {
            float sy = dy * scaleY;
            for (int dx = 0; dx < dstW; dx++)
            {
                float sx = dx * scaleX;
                float r = SampleBilinearChannel(srcRgb, srcW, srcH, sx, sy, 0);
                float g = SampleBilinearChannel(srcRgb, srcW, srcH, sx, sy, 1);
                float b = SampleBilinearChannel(srcRgb, srcW, srcH, sx, sy, 2);
                result[dy * dstW + dx] = 0.299f * r + 0.587f * g + 0.114f * b;
            }
        }

        return result;
    }

    private static float SampleBilinearChannel(byte[] srcRgb, int srcW, int srcH, float x, float y, int channel)
    {
        int x0 = Math.Clamp((int)MathF.Floor(x), 0, srcW - 1);
        int y0 = Math.Clamp((int)MathF.Floor(y), 0, srcH - 1);
        int x1 = Math.Min(x0 + 1, srcW - 1);
        int y1 = Math.Min(y0 + 1, srcH - 1);

        float fx = Math.Clamp(x - x0, 0f, 1f);
        float fy = Math.Clamp(y - y0, 0f, 1f);

        float v00 = srcRgb[(y0 * srcW + x0) * 3 + channel];
        float v10 = srcRgb[(y0 * srcW + x1) * 3 + channel];
        float v01 = srcRgb[(y1 * srcW + x0) * 3 + channel];
        float v11 = srcRgb[(y1 * srcW + x1) * 3 + channel];

        float top = (1f - fx) * v00 + fx * v10;
        float bottom = (1f - fx) * v01 + fx * v11;
        return (1f - fy) * top + fy * bottom;
    }

    private static float[] Softmax(float[] values)
    {
        float max = values.Max();
        var exp = new float[values.Length];
        float sum = 0f;

        for (int i = 0; i < values.Length; i++)
        {
            exp[i] = MathF.Exp(values[i] - max);
            sum += exp[i];
        }

        if (sum <= 0f)
            return new float[values.Length];

        for (int i = 0; i < exp.Length; i++)
            exp[i] /= sum;

        return exp;
    }

    private static ModelConfig GetModelConfig(InferenceSession session)
    {
        var inputMetadata = session.InputMetadata.First();
        int inputHeight = ResolveDimension(inputMetadata.Value.Dimensions, 2, 64);
        int inputWidth = ResolveDimension(inputMetadata.Value.Dimensions, 3, 64);
        int channelCount = ResolveDimension(inputMetadata.Value.Dimensions, 1, 1);

        if (channelCount != 1)
            throw new NotSupportedException($"FaceEmotion.cs currently supports only single-channel grayscale input models. Actual channel count: {channelCount}.");

        bool hasEmotionOutput = session.OutputMetadata.Any(item => FlattenedLength(item.Value.Dimensions) == DefaultLabels.Length);
        if (!hasEmotionOutput)
            throw new NotSupportedException($"FaceEmotion.cs currently supports only ONNX models with {DefaultLabels.Length} emotion outputs.");

        return new ModelConfig(inputMetadata.Key, inputWidth, inputHeight, DefaultLabels);
    }

    private static int FlattenedLength(IReadOnlyList<int> dimensions)
    {
        int length = 1;
        bool hasKnown = false;

        foreach (int dimension in dimensions)
        {
            if (dimension <= 0)
                continue;

            hasKnown = true;
            length *= dimension;
        }

        return hasKnown ? length : 0;
    }

    private static int ResolveDimension(IReadOnlyList<int> dimensions, int index, int fallback)
    {
        if (index < dimensions.Count && dimensions[index] > 0)
            return dimensions[index];

        return fallback;
    }

    private static void DrawRectOutline(
        byte[] pixels,
        int imageWidth,
        int imageHeight,
        FaceLandmarkBox box,
        int thickness,
        byte r,
        byte g,
        byte b)
    {
        int xmin = Math.Clamp((int)MathF.Round(box.Xmin), 0, imageWidth - 1);
        int ymin = Math.Clamp((int)MathF.Round(box.Ymin), 0, imageHeight - 1);
        int xmax = Math.Clamp((int)MathF.Round(box.Xmax), 0, imageWidth - 1);
        int ymax = Math.Clamp((int)MathF.Round(box.Ymax), 0, imageHeight - 1);

        for (int t = 0; t < thickness; t++)
        {
            for (int x = xmin; x <= xmax; x++) SetPixel(pixels, imageWidth, imageHeight, x, ymin + t, r, g, b, 255);
            for (int x = xmin; x <= xmax; x++) SetPixel(pixels, imageWidth, imageHeight, x, ymax - t, r, g, b, 255);
            for (int y = ymin; y <= ymax; y++) SetPixel(pixels, imageWidth, imageHeight, xmin + t, y, r, g, b, 255);
            for (int y = ymin; y <= ymax; y++) SetPixel(pixels, imageWidth, imageHeight, xmax - t, y, r, g, b, 255);
        }
    }

    //private static void DrawLabelText(
    //    byte[] pixels,
    //    int imageWidth,
    //    int imageHeight,
    //    int anchorX,
    //    int anchorY,
    //    string text,
    //    Season.Fonts.Font font,
    //    int fontSize,
    //    byte r,
    //    byte g,
    //    byte b)
    //{
    //    if (string.IsNullOrWhiteSpace(text))
    //        return;

    //    int textHeight = 0;
    //    var glyphs = new List<(byte[] Buffer, int Width, int Height)>();

    //    foreach (var ch in text)
    //    {
    //        var glyph = font.CreateGlyph(fontSize, ch, stroke: false);
    //        if (glyph.colorBuffer == null || glyph.glyphMetrics.Width <= 0 || glyph.glyphMetrics.Height <= 0)
    //            continue;

    //        glyphs.Add((glyph.colorBuffer, glyph.glyphMetrics.Width, glyph.glyphMetrics.Height));
    //        textHeight = Math.Max(textHeight, glyph.glyphMetrics.Height);
    //    }

    //    if (glyphs.Count == 0)
    //        return;

    //    int textWidth = glyphs.Sum(g => g.Width + 1);
    //    int boxWidth = textWidth + 4;
    //    int boxHeight = textHeight + 4;
    //    int boxX = Math.Clamp(anchorX, 0, Math.Max(0, imageWidth - boxWidth));
    //    int preferredY = anchorY - boxHeight - 2;
    //    int boxY = preferredY >= 0 ? preferredY : Math.Min(Math.Max(0, anchorY + 2), Math.Max(0, imageHeight - boxHeight));

    //    FillRect(pixels, imageWidth, imageHeight, boxX, boxY, boxWidth, boxHeight, 0, 0, 0, 180);

    //    int cursorX = boxX + 2;
    //    foreach (var (buffer, width, height) in glyphs)
    //    {
    //        BlendGlyph(pixels, imageWidth, imageHeight, cursorX, boxY + 2, buffer, width, height, r, g, b);
    //        cursorX += width + 1;
    //    }
    //}

    private static void SetPixel(byte[] pixels, int imageWidth, int imageHeight, int x, int y, byte r, byte g, byte b, byte a)
    {
        if ((uint)x >= imageWidth || (uint)y >= imageHeight)
            return;

        int idx = (y * imageWidth + x) * 4;
        pixels[idx] = r;
        pixels[idx + 1] = g;
        pixels[idx + 2] = b;
        pixels[idx + 3] = a;
    }

    private static void FillRect(byte[] pixels, int imageWidth, int imageHeight, int x, int y, int width, int height, byte r, byte g, byte b, byte alpha)
    {
        int startX = Math.Max(x, 0);
        int startY = Math.Max(y, 0);
        int endX = Math.Min(x + width, imageWidth);
        int endY = Math.Min(y + height, imageHeight);

        for (int py = startY; py < endY; py++)
        {
            for (int px = startX; px < endX; px++)
            {
                int idx = (py * imageWidth + px) * 4;
                float a = alpha / 255f;
                float inv = 1f - a;
                pixels[idx] = (byte)Math.Clamp((int)MathF.Round(r * a + pixels[idx] * inv), 0, 255);
                pixels[idx + 1] = (byte)Math.Clamp((int)MathF.Round(g * a + pixels[idx + 1] * inv), 0, 255);
                pixels[idx + 2] = (byte)Math.Clamp((int)MathF.Round(b * a + pixels[idx + 2] * inv), 0, 255);
                pixels[idx + 3] = 255;
            }
        }
    }

    private static void BlendGlyph(byte[] pixels, int imageWidth, int imageHeight, int x, int y, byte[] glyphBuffer, int glyphWidth, int glyphHeight, byte r, byte g, byte b)
    {
        for (int gy = 0; gy < glyphHeight; gy++)
        {
            for (int gx = 0; gx < glyphWidth; gx++)
            {
                int glyphIndex = (gy * glyphWidth + gx) * 4;
                byte alpha = glyphBuffer[glyphIndex + 3];
                if (alpha == 0)
                    continue;

                int px = x + gx;
                int py = y + gy;
                if ((uint)px >= imageWidth || (uint)py >= imageHeight)
                    continue;

                int idx = (py * imageWidth + px) * 4;
                float a = alpha / 255f;
                float inv = 1f - a;
                pixels[idx] = (byte)Math.Clamp((int)MathF.Round(r * a + pixels[idx] * inv), 0, 255);
                pixels[idx + 1] = (byte)Math.Clamp((int)MathF.Round(g * a + pixels[idx + 1] * inv), 0, 255);
                pixels[idx + 2] = (byte)Math.Clamp((int)MathF.Round(b * a + pixels[idx + 2] * inv), 0, 255);
                pixels[idx + 3] = 255;
            }
        }
    }

    private sealed record ModelConfig(string InputName, int InputWidth, int InputHeight, IReadOnlyList<string> Labels);
    private sealed record FaceSquareCrop(int X, int Y, int Size);
    private sealed record DecodedEmotion(string Emotion, float EmotionConfidence, List<FaceEmotionScore> Scores);
}

public sealed class FaceEmotionResult
{
    public string Model { get; set; } = string.Empty;
    public string DetectorModel { get; set; } = string.Empty;
    public int ImageWidth { get; set; }
    public int ImageHeight { get; set; }
    public int RequestedMaxFaces { get; set; }
    public int InputWidth { get; set; }
    public int InputHeight { get; set; }
    public string[] EmotionLabels { get; set; } = [];
    public List<FaceEmotionFace> Faces { get; set; } = new();
    public byte[] AnnotatedImage { get; set; } = [];

    public string Summary
    {
        get
        {
            return String.Join("\r\n", Faces.Select(fa => $"Index:{fa.Index} Confidence:{fa.DetectionConfidence} Emotion:{fa.Emotion} EmotionConfidence:{fa.EmotionConfidence} BoundingBox:{fa.BoundingBox.Xmin} {fa.BoundingBox.Ymin} {fa.BoundingBox.Xmax} {fa.BoundingBox.Ymax} CropBox:{fa.CropBox.Xmin} {fa.CropBox.Ymin} {fa.CropBox.Xmax} {fa.CropBox.Ymax} Scores:{fa.Scores.Count}"));
        }
    }
}

public sealed class FaceEmotionFace
{
    public int Index { get; set; }
    public float DetectionConfidence { get; set; }
    public FaceLandmarkBox BoundingBox { get; set; } = null!;
    public FaceLandmarkBox CropBox { get; set; } = null!;
    public string Emotion { get; set; } = string.Empty;
    public float EmotionConfidence { get; set; }
    public List<FaceEmotionScore> Scores { get; set; } = new();
}

public sealed class FaceEmotionScore
{
    public string Label { get; set; } = string.Empty;
    public float Score { get; set; }
}
