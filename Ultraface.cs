// Copyright (c) SeasonEngine and contributors.
// Licensed under the MIT License.
// https://github.com/SeasonRealms/SeasonVision

namespace SeasonVision;

/// <summary>
/// Ultraface face detection inference.
/// Matches the MauiVisionSample preprocessing flow: resize directly to 320x240
/// and normalize with (pixel - 127) / 128.
/// </summary>
public static class Ultraface
{
    private const int InputWidth = 320;
    private const int InputHeight = 240;
    private const int DisplayShortestEdge = 800;
    private const float MinConfidence = 0.5f;
    private const float NmsIouThreshold = 0.3f;
    private const int FontSize = 16;

    public static UltrafaceFaceResult Detect(string model, ReadOnlySpan<byte> imageData,
        int width,
        int height, bool createAnnotatedImage = false, int maxFaces = int.MaxValue)
    {
        var result = new UltrafaceFaceResult();

        var displayImage = ImageProcessor.ResizeShortestEdge(imageData, width, height, DisplayShortestEdge);
        var faces = DetectFaces(model, imageData, width, height, maxFaces);

        if (createAnnotatedImage)
        {
            if (faces.Count == 0)
            {
                result.AnnotatedImage = imageData.ToArray();
            }
            else
            {
                var pixels = ImageProcessor.EnsureRgba(imageData, width, height);
                //var font = new Season.Fonts.Font("Sample/Ravie.ttf", FontSize, false);
                float scaleX = (float)width / width;
                float scaleY = (float)height / height;

                foreach (var face in faces)
                {
                    DrawPrediction(
                        pixels,
                        width,
                        height,
                        new UltrafaceFace
                        {
                            Confidence = face.Confidence,
                            Box = face.Box.Scale(scaleX, scaleY)
                        });
                }

                result.AnnotatedImage = pixels;
            }
        }

        result.Faces = faces;

        return result;
    }

    static List<UltrafaceFace> DetectFaces(string model, ReadOnlySpan<byte> imageData,
        int width,
        int height, 
        int maxFaces = int.MaxValue)
    {
        var rgb = ImageProcessor.ExtractRgb(imageData, width, height);
        var mean = new[] { 127.0f, 127.0f, 127.0f };
        var stddev = new[] { 128.0f, 128.0f, 128.0f };

        float[] chw = ImageProcessor.ResizeNormalizeToNchw(
            rgb,
            width,
            height,
            InputWidth,
            InputHeight,
            mean,
            stddev,
            scaleToUnitInterval: false);

        var input = new DenseTensor<float>(chw, new[] { 1, 3, InputHeight, InputWidth });
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input", input)
        };

        using var session = new InferenceSession(model);
        using var results = session.Run(inputs);
        var resultsArray = results.ToArray();
        float[] confidences = resultsArray[0].AsEnumerable<float>().ToArray();
        float[] boxes = resultsArray[1].AsEnumerable<float>().ToArray();

        return GetPredictions(confidences, boxes, width, height, maxFaces);
    }

    private static List<UltrafaceFace> GetPredictions(float[] confidences, float[] boxes, int imageWidth, int imageHeight, int maxFaces)
    {
        var predictions = new List<UltrafaceFace>();

        for (int i = 1, scoreIndex = 0; i < confidences.Length; i += 2, scoreIndex++)
        {
            float confidence = confidences[i];
            if (confidence < MinConfidence)
                continue;

            int boxOffset = scoreIndex * 4;
            if (boxOffset + 3 >= boxes.Length)
                continue;

            float xmin = boxes[boxOffset] * imageWidth;
            float ymin = boxes[boxOffset + 1] * imageHeight;
            float xmax = boxes[boxOffset + 2] * imageWidth;
            float ymax = boxes[boxOffset + 3] * imageHeight;

            if (xmax <= xmin || ymax <= ymin)
                continue;

            predictions.Add(new UltrafaceFace
            {
                Confidence = confidence,
                Box = new UltrafaceBox(xmin, ymin, xmax, ymax)
            });
        }

        var kept = ApplyNms(predictions, NmsIouThreshold);
        if (maxFaces > 0 && kept.Count > maxFaces)
            kept = kept.Take(maxFaces).ToList();

        return kept;
    }

    private static List<UltrafaceFace> ApplyNms(List<UltrafaceFace> predictions, float iouThreshold)
    {
        if (predictions.Count <= 1)
            return predictions;

        var ordered = predictions
            .OrderByDescending(prediction => prediction.Confidence)
            .ToList();

        var kept = new List<UltrafaceFace>(ordered.Count);

        foreach (var candidate in ordered)
        {
            bool suppress = false;

            foreach (var keptPrediction in kept)
            {
                if (ComputeIoU(candidate.Box, keptPrediction.Box) > iouThreshold)
                {
                    suppress = true;
                    break;
                }
            }

            if (!suppress)
                kept.Add(candidate);
        }

        return kept;
    }

    private static float ComputeIoU(UltrafaceBox a, UltrafaceBox b)
    {
        float interXmin = Math.Max(a.Xmin, b.Xmin);
        float interYmin = Math.Max(a.Ymin, b.Ymin);
        float interXmax = Math.Min(a.Xmax, b.Xmax);
        float interYmax = Math.Min(a.Ymax, b.Ymax);

        float interWidth = Math.Max(0f, interXmax - interXmin);
        float interHeight = Math.Max(0f, interYmax - interYmin);
        float interArea = interWidth * interHeight;

        float areaA = (a.Xmax - a.Xmin) * (a.Ymax - a.Ymin);
        float areaB = (b.Xmax - b.Xmin) * (b.Ymax - b.Ymin);
        float unionArea = areaA + areaB - interArea;

        if (unionArea <= 0f)
            return 0f;

        return interArea / unionArea;
    }

    private static void DrawPrediction(byte[] pixels, int imageWidth, int imageHeight, UltrafaceFace prediction)
    {
        int xmin = (int)prediction.Box.Xmin;
        int ymin = (int)prediction.Box.Ymin;
        int xmax = (int)prediction.Box.Xmax;
        int ymax = (int)prediction.Box.Ymax;
        int thickness = Math.Max(2, Math.Min(imageWidth, imageHeight) / 300);

        DrawRectOutline(pixels, imageWidth, imageHeight, xmin, ymin, xmax, ymax, thickness, 255, 0, 0);

        //var text = $"{prediction.Confidence:P2}";
        //DrawLabelText(pixels, imageWidth, imageHeight, xmin, ymin, text, font, fontSize, 255, 255, 255);
    }

    private static void DrawRectOutline(
        byte[] pixels, int imageWidth, int imageHeight,
        int xmin, int ymin, int xmax, int ymax,
        int thickness, byte r, byte g, byte b)
    {
        for (int t = 0; t < thickness; t++)
        {
            for (int x = xmin; x <= xmax; x++) SetPixel(pixels, imageWidth, imageHeight, x, ymin + t, r, g, b, 255);
            for (int x = xmin; x <= xmax; x++) SetPixel(pixels, imageWidth, imageHeight, x, ymax - t, r, g, b, 255);
            for (int y = ymin; y <= ymax; y++) SetPixel(pixels, imageWidth, imageHeight, xmin + t, y, r, g, b, 255);
            for (int y = ymin; y <= ymax; y++) SetPixel(pixels, imageWidth, imageHeight, xmax - t, y, r, g, b, 255);
        }
    }

    //private static void DrawLabelText(
    //    byte[] pixels, int imageWidth, int imageHeight,
    //    int anchorX, int anchorY, string text,
    //    Season.Fonts.Font font, int fontSize,
    //    byte r, byte g, byte b)
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
                pixels[idx] = (byte)Math.Clamp((int)Math.Round(r * a + pixels[idx] * inv), 0, 255);
                pixels[idx + 1] = (byte)Math.Clamp((int)Math.Round(g * a + pixels[idx + 1] * inv), 0, 255);
                pixels[idx + 2] = (byte)Math.Clamp((int)Math.Round(b * a + pixels[idx + 2] * inv), 0, 255);
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
                pixels[idx] = (byte)Math.Clamp((int)Math.Round(r * a + pixels[idx] * inv), 0, 255);
                pixels[idx + 1] = (byte)Math.Clamp((int)Math.Round(g * a + pixels[idx + 1] * inv), 0, 255);
                pixels[idx + 2] = (byte)Math.Clamp((int)Math.Round(b * a + pixels[idx + 2] * inv), 0, 255);
                pixels[idx + 3] = 255;
            }
        }
    }
}

public sealed class UltrafaceFace
{
    public UltrafaceBox Box { get; set; } = null!;
    public float Confidence { get; set; }
}

public sealed class UltrafaceBox
{
    public float Xmin { get; }
    public float Ymin { get; }
    public float Xmax { get; }
    public float Ymax { get; }

    public float Width => Xmax - Xmin;
    public float Height => Ymax - Ymin;

    public UltrafaceBox(float xmin, float ymin, float xmax, float ymax)
    {
        Xmin = xmin;
        Ymin = ymin;
        Xmax = xmax;
        Ymax = ymax;
    }

    public UltrafaceBox Scale(float scaleX, float scaleY)
    {
        return new UltrafaceBox(
            Xmin * scaleX,
            Ymin * scaleY,
            Xmax * scaleX,
            Ymax * scaleY);
    }
}

public sealed class UltrafaceFaceResult
{
    public List<UltrafaceFace> Faces { get; set; } = new();

    public byte[] AnnotatedImage { get; set; } = [];
}
