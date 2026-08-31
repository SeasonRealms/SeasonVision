// Copyright (c) SeasonEngine and contributors.
// Licensed under the MIT License.
// https://github.com/SeasonRealms/SeasonVision

namespace Season.Vision;

/// <summary>
/// Face attribute analysis.
/// The current implementation follows the input and output contract of
/// OpenVINO age-gender-recognition-retail-0013 using the converted ONNX model
/// directly through OnnxRuntime.
/// </summary>
public static class FaceAttributes
{
    public const string DefaultModel = "Sample/age-gender-recognition-retail-0013.onnx";
    public const string DefaultDetectorModel = "Sample/Ultraface_version-RFB-320.onnx";

    private const float CropPadding = 0.08f;
    private const int FontSize = 16;

    public static FaceAttributesResult Detect(
        InferenceSession detectorSession,
        InferenceSession recognizerSession,
        ReadOnlySpan<byte> imageData,
        int width,
        int height,
        bool createAnnotatedImage = false,
        int maxFaces = 5)
    {
        if (imageData == Span<byte>.Empty)
        {
            throw new ArgumentNullException(nameof(imageData));
        }
        
        if (width <= 0)
            throw new ArgumentOutOfRangeException(nameof(width), "Width must be greater than 0.");

        if (height <= 0)
            throw new ArgumentOutOfRangeException(nameof(height), "Height must be greater than 0.");

        if (maxFaces <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxFaces), "maxFaces must be greater than 0.");

        //string modelPath = DeviceServices.Core.LoadFilePath(model);
        //using var session = new InferenceSession(model);
        var config = GetModelConfig(recognizerSession);
        var detectorFaces = Ultraface.Detect(detectorSession, imageData, width, height, false, maxFaces);
        var rgb = ImageProcessor.ExtractRgb(imageData, width, height);

        var result = new FaceAttributesResult
        {
            //Model = model,
            //DetectorModel = null, //detectorModel,
            ImageWidth = width,
            ImageHeight = height,
            RequestedMaxFaces = maxFaces,
            InputWidth = config.InputWidth,
            InputHeight = config.InputHeight,
            GenderLabels = ["Female", "Male"]
        };

        foreach (var detectorFace in detectorFaces.Faces)
        {
            var crop = ExpandCrop(detectorFace.Box, width, height);
            var cropRgb = ImageProcessor.CropRgb(rgb, width, height, crop.X, crop.Y, crop.Width, crop.Height);
            var chw = ResizeToBgrNchw(cropRgb, crop.Width, crop.Height, config.InputWidth, config.InputHeight);

            var input = new DenseTensor<float>(chw, new[] { 1, 3, config.InputHeight, config.InputWidth });
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(config.InputName, input)
            };

            using var outputs = recognizerSession.Run(inputs);
            var attribute = Decode(outputs, config);

            result.Faces.Add(new FaceAttributeFace
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
                    crop.X + crop.Width - 1,
                    crop.Y + crop.Height - 1),
                Age = attribute.Age,
                Gender = attribute.Gender,
                GenderConfidence = attribute.GenderConfidence,
                FemaleScore = attribute.FemaleScore,
                MaleScore = attribute.MaleScore
            });
        }

        if (createAnnotatedImage)
            result.AnnotatedImage = DrawResults(imageData, width, height, result.Faces);

        return result;
    }

    private static float[] ResizeToBgrNchw(byte[] srcRgb, int srcW, int srcH, int dstW, int dstH)
    {
        float scaleX = (float)srcW / dstW;
        float scaleY = (float)srcH / dstH;
        float[] result = new float[3 * dstW * dstH];

        for (int dy = 0; dy < dstH; dy++)
        {
            float sy = dy * scaleY;
            for (int dx = 0; dx < dstW; dx++)
            {
                float sx = dx * scaleX;
                int dstIdx = dy * dstW + dx;

                float r = SampleBilinearChannel(srcRgb, srcW, srcH, sx, sy, 0);
                float g = SampleBilinearChannel(srcRgb, srcW, srcH, sx, sy, 1);
                float b = SampleBilinearChannel(srcRgb, srcW, srcH, sx, sy, 2);

                result[dstIdx] = b;
                result[dstW * dstH + dstIdx] = g;
                result[2 * dstW * dstH + dstIdx] = r;
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

    private static DecodedAttributes Decode(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs,
        ModelConfig config)
    {
        var tensors = outputs.ToArray();
        if (tensors.Length < 2)
            throw new InvalidOperationException($"Unexpected number of FaceAttributes output tensors. Expected at least 2, actual: {tensors.Length}.");

        DisposableNamedOnnxValue? ageTensor = null;
        DisposableNamedOnnxValue? genderTensor = null;

        foreach (var tensor in tensors)
        {
            var dims = tensor.AsTensor<float>().Dimensions.ToArray();
            if (dims.Length >= 2 && dims[1] == 1)
                ageTensor ??= tensor;
            else if (dims.Length >= 2 && dims[1] == 2)
                genderTensor ??= tensor;
        }

        if (ageTensor == null || genderTensor == null)
            throw new NotSupportedException("Only FaceAttributes ONNX models with one age scalar output and one two-class gender output are supported.");

        float ageRaw = ageTensor.AsEnumerable<float>().FirstOrDefault();
        float[] genderLogits = genderTensor.AsEnumerable<float>().ToArray();
        if (genderLogits.Length < 2)
            throw new InvalidOperationException("Unexpected gender output length.");

        var genderProbabilities = Softmax(genderLogits);
        bool isMale = genderProbabilities[1] >= genderProbabilities[0];

        return new DecodedAttributes(
            Age: ageRaw * config.AgeScale,
            Gender: isMale ? "Male" : "Female",
            GenderConfidence: Math.Max(genderProbabilities[0], genderProbabilities[1]),
            FemaleScore: genderProbabilities[0],
            MaleScore: genderProbabilities[1]);
    }

    private static byte[] DrawResults(ReadOnlySpan<byte> imageData,
        int width,
        int height, List<FaceAttributeFace> faces)
    {
        if (faces.Count == 0)
            return imageData.ToArray();

        var pixels = ImageProcessor.EnsureRgba(imageData, width, height);
        int thickness = Math.Max(2, Math.Min(width, height) / 300);

        foreach (var face in faces)
        {
            DrawRectOutline(pixels, width, height, face.BoundingBox, thickness, 255, 0, 0);
            //string text = $"{face.Gender} {face.GenderConfidence:P0}, {MathF.Round(face.Age)}y";
            //DrawLabelText(
            //    pixels,
            //    imageResult.Width,
            //    imageResult.Height,
            //    (int)face.BoundingBox.Xmin,
            //    (int)face.BoundingBox.Ymin,
            //    text,
            //    font,
            //    FontSize,
            //    255,
            //    255,
            //    255);
        }

        return pixels;
    }

    private static FaceCrop ExpandCrop(UltrafaceBox box, int imageWidth, int imageHeight)
    {
        float faceWidth = box.Xmax - box.Xmin + 1f;
        float faceHeight = box.Ymax - box.Ymin + 1f;

        int x1 = Math.Max(0, (int)(box.Xmin - faceWidth * CropPadding));
        int y1 = Math.Max(0, (int)(box.Ymin - faceHeight * CropPadding));
        int x2 = Math.Min(imageWidth - 1, (int)(box.Xmax + faceWidth * CropPadding));
        int y2 = Math.Min(imageHeight - 1, (int)(box.Ymax + faceHeight * CropPadding));

        return new FaceCrop(
            x1,
            y1,
            Math.Max(1, x2 - x1 + 1),
            Math.Max(1, y2 - y1 + 1));
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
        int inputHeight = ResolveDimension(inputMetadata.Value.Dimensions, 2, 62);
        int inputWidth = ResolveDimension(inputMetadata.Value.Dimensions, 3, 62);

        bool hasAgeOutput = session.OutputMetadata.Any(item => ResolveDimension(item.Value.Dimensions, 1, 0) == 1);
        bool hasGenderOutput = session.OutputMetadata.Any(item => ResolveDimension(item.Value.Dimensions, 1, 0) == 2);

        if (!hasAgeOutput || !hasGenderOutput)
            throw new NotSupportedException("FaceAttributes.cs currently supports only ONNX models with a 1-channel age output and a 2-channel gender output.");

        return new ModelConfig(inputMetadata.Key, inputWidth, inputHeight, AgeScale: 100f);
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

    private sealed record ModelConfig(string InputName, int InputWidth, int InputHeight, float AgeScale);
    private sealed record FaceCrop(int X, int Y, int Width, int Height);
    private sealed record DecodedAttributes(float Age, string Gender, float GenderConfidence, float FemaleScore, float MaleScore);
}

public sealed class FaceAttributesResult
{
    public string Model { get; set; } = string.Empty;
    public string DetectorModel { get; set; } = string.Empty;
    public int ImageWidth { get; set; }
    public int ImageHeight { get; set; }
    public int RequestedMaxFaces { get; set; }
    public int InputWidth { get; set; }
    public int InputHeight { get; set; }
    public string[] GenderLabels { get; set; } = [];
    public List<FaceAttributeFace> Faces { get; set; } = new();
    public byte[] AnnotatedImage { get; set; } = [];

    public string Summary
    {
        get
        {
            return String.Join("\r\n", Faces.Select(fa => $"Index:{fa.Index} Confidence:{fa.DetectionConfidence} Age:{fa.Age} Gender:{fa.Gender} GenderConfidence:{fa.GenderConfidence} FemaleScore:{fa.FemaleScore} MaleScore:{fa.MaleScore} BoundingBox:{fa.BoundingBox.Xmin} {fa.BoundingBox.Ymin} {fa.BoundingBox.Xmax} {fa.BoundingBox.Ymax} CropBox:{fa.CropBox.Xmin} {fa.CropBox.Ymin} {fa.CropBox.Xmax} {fa.CropBox.Ymax}"));
        }
    }
}

public sealed class FaceAttributeFace
{
    public int Index { get; set; }
    public float DetectionConfidence { get; set; }
    public FaceLandmarkBox BoundingBox { get; set; } = null!;
    public FaceLandmarkBox CropBox { get; set; } = null!;
    public float Age { get; set; }
    public string Gender { get; set; } = string.Empty;
    public float GenderConfidence { get; set; }
    public float FemaleScore { get; set; }
    public float MaleScore { get; set; }
}
