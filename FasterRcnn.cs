// Copyright (c) SeasonEngine and contributors.
// Licensed under the MIT License.
// https://github.com/SeasonRealms/SeasonVision

using static SeasonVision.MaskRcnn;

namespace SeasonVision;

public class FasterRcnn
{
    public static readonly string[] Labels = new[] {"__background",
                                                        "person",
                                                        "bicycle",
                                                        "car",
                                                        "motorcycle",
                                                        "airplane",
                                                        "bus",
                                                        "train",
                                                        "truck",
                                                        "boat",
                                                        "traffic light",
                                                        "fire hydrant",
                                                        "stop sign",
                                                        "parking meter",
                                                        "bench",
                                                        "bird",
                                                        "cat",
                                                        "dog",
                                                        "horse",
                                                        "sheep",
                                                        "cow",
                                                        "elephant",
                                                        "bear",
                                                        "zebra",
                                                        "giraffe",
                                                        "backpack",
                                                        "umbrella",
                                                        "handbag",
                                                        "tie",
                                                        "suitcase",
                                                        "frisbee",
                                                        "skis",
                                                        "snowboard",
                                                        "sports ball",
                                                        "kite",
                                                        "baseball bat",
                                                        "baseball glove",
                                                        "skateboard",
                                                        "surfboard",
                                                        "tennis racket",
                                                        "bottle",
                                                        "wine glass",
                                                        "cup",
                                                        "fork",
                                                        "knife",
                                                        "spoon",
                                                        "bowl",
                                                        "banana",
                                                        "apple",
                                                        "sandwich",
                                                        "orange",
                                                        "broccoli",
                                                        "carrot",
                                                        "hot dog",
                                                        "pizza",
                                                        "donut",
                                                        "cake",
                                                        "chair",
                                                        "couch",
                                                        "potted plant",
                                                        "bed",
                                                        "dining table",
                                                        "toilet",
                                                        "tv",
                                                        "laptop",
                                                        "mouse",
                                                        "remote",
                                                        "keyboard",
                                                        "cell phone",
                                                        "microwave",
                                                        "oven",
                                                        "toaster",
                                                        "sink",
                                                        "refrigerator",
                                                        "book",
                                                        "clock",
                                                        "vase",
                                                        "scissors",
                                                        "teddy bear",
                                                        "hair drier",
                                                        "toothbrush"};

    /// <summary>
    /// Faster R-CNN object detection for the 80 COCO categories.
    /// Preprocessing resizes the shorter edge to 800, pads to a multiple of 32,
    /// reorders channels to BGR, and subtracts the dataset mean.
    /// Returns predictions and can optionally include an annotated RGBA image.
    /// </summary>
    /// <param name="model">Path to the ONNX model file, for example "Sample/FasterRCNN-12-qdq.onnx".</param>
    /// <param name="imageData">RGBA image bytes.</param>
    /// <returns>A result object containing predictions and an optional annotated image.</returns>
    public static FasterRcnnResult Detect(string model, ReadOnlySpan<byte> imageData, int width, int height, bool createAnnotatedImage = false)
    {
        var result = new FasterRcnnResult();

        const int shortSide = 800;
        const int alignment = 32;
        const float minConfidence = 0.7f;

        // Preprocess the input image.
        var rgb = ImageProcessor.ExtractRgb(imageData, width, height);
        var mean = new[] { 102.9801f, 115.9465f, 122.7717f };
        float[] chw = ImageProcessor.ResizePadBgrNormalize(
            rgb, width, height, shortSide, alignment, mean);

        // Compute the padded tensor size using the same rules as ImageProcessor.
        float scale = (float)shortSide / Math.Min(width, height);
        int newW = Math.Max(1, (int)(width * scale));
        int newH = Math.Max(1, (int)(height * scale));
        int paddedW = ((newW + alignment - 1) / alignment) * alignment;
        int paddedH = ((newH + alignment - 1) / alignment) * alignment;

        // Build a [3, paddedH, paddedW] tensor without a batch dimension.
        var input = new DenseTensor<float>(chw, new[] { 3, paddedH, paddedW });

        // Run ONNX inference.
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("image", input)
        };

        using var session = new InferenceSession(model);
        using var results = session.Run(inputs);
        var resultsArray = results.ToArray();

        float[] boxes = resultsArray[0].AsEnumerable<float>().ToArray();
        long[] labels = resultsArray[1].AsEnumerable<long>().ToArray();
        float[] confidences = resultsArray[2].AsEnumerable<float>().ToArray();

        // Collect high-confidence predictions.
        var predictions = new List<Prediction>();
        float invScale = 1f / scale;
        for (int i = 0; i + 3 < boxes.Length; i += 4)
        {
            int idx = i / 4;
            if (idx >= confidences.Length || confidences[idx] < minConfidence)
                continue;

            int labelIdx = (int)labels[idx];
            string label = labelIdx > 0 && labelIdx < Labels.Length
                ? Labels[labelIdx]
                : $"cls_{labelIdx}";

            // Map model-space boxes back to the original image coordinates.
            predictions.Add(new Prediction
            {
                Box = new Box(
                    boxes[i] * invScale,
                    boxes[i + 1] * invScale,
                    boxes[i + 2] * invScale,
                    boxes[i + 3] * invScale),
                Label = label,
                Confidence = confidences[idx]
            });
        }

        result.Predictions = predictions;

        if (createAnnotatedImage)
        {
            // imageData is already RGBA, so copy it directly.
            int pixelCount = width * height;
            var srcData = imageData;
            var pixels = new byte[pixelCount * 4];
            for (int i = 0; i < pixelCount; i++)
            {
                int s = i * 4;
                int d = i * 4;
                pixels[d] = srcData[s];           // R
                pixels[d + 1] = srcData[s + 1];   // G
                pixels[d + 2] = srcData[s + 2];   // B
                pixels[d + 3] = srcData[s + 3];   // A
            }

            if (predictions.Count > 0)
            {
                //var font = new Season.Fonts.Font("Sample/Ravie.ttf", fontSize, false);

                foreach (var p in predictions)
                {
                    DrawBox(pixels, width, height, p);
                }
            }

            result.AnnotatedImage = pixels;
        }

        return result;
    }

    // Drawing helpers.

    private static void DrawBox(
        byte[] pixels, int w, int h, Prediction p)
    {
        int xmin = (int)p.Box.Xmin;
        int ymin = (int)p.Box.Ymin;
        int xmax = (int)p.Box.Xmax;
        int ymax = (int)p.Box.Ymax;
        int thickness = Math.Max(2, Math.Min(w, h) / 300);

        // Draw the detection box in red.
        DrawRectOutline(pixels, w, h, xmin, ymin, xmax, ymax, thickness, 255, 0, 0);

        // Draw the label text in white.
        //var text = $"{p.Label}, {p.Confidence:0.00}";
        //DrawLabelText(pixels, w, h, xmin, ymin, text, font, fontSize, 255, 255, 255);
    }

    private static void DrawRectOutline(
        byte[] pixels, int w, int h,
        int xmin, int ymin, int xmax, int ymax,
        int thickness, byte r, byte g, byte b)
    {
        for (int t = 0; t < thickness; t++)
        {
            // Top edge
            for (int x = xmin; x <= xmax; x++) SetPixel(pixels, w, h, x, ymin + t, r, g, b, 255);
            // Bottom edge
            for (int x = xmin; x <= xmax; x++) SetPixel(pixels, w, h, x, ymax - t, r, g, b, 255);
            // Left edge
            for (int y = ymin; y <= ymax; y++) SetPixel(pixels, w, h, xmin + t, y, r, g, b, 255);
            // Right edge
            for (int y = ymin; y <= ymax; y++) SetPixel(pixels, w, h, xmax - t, y, r, g, b, 255);
        }
    }

    //private static void DrawLabelText(
    //    byte[] pixels, int imageW, int imageH,
    //    int anchorX, int anchorY, string text,
    //    Season.Fonts.Font font, int fontSize,
    //    byte r, byte g, byte b)
    //{
    //    if (string.IsNullOrWhiteSpace(text))
    //        return;

    //    int textH = 0;
    //    var glyphs = new List<(byte[] Buffer, int W, int H)>();

    //    foreach (var ch in text)
    //    {
    //        var glyph = font.CreateGlyph(fontSize, ch, stroke: false);
    //        if (glyph.colorBuffer == null || glyph.glyphMetrics.Width <= 0 || glyph.glyphMetrics.Height <= 0)
    //            continue;

    //        glyphs.Add((glyph.colorBuffer, glyph.glyphMetrics.Width, glyph.glyphMetrics.Height));
    //        textH = Math.Max(textH, glyph.glyphMetrics.Height);
    //    }

    //    if (glyphs.Count == 0)
    //        return;

    //    // Draw text on a translucent background for readability.
    //    int textW = glyphs.Sum(g => g.W + 1);
    //    int boxW = textW + 4;
    //    int boxH = textH + 4;

    //    int boxX = Math.Clamp(anchorX, 0, Math.Max(0, imageW - boxW));
    //    int preferredY = anchorY - boxH - 2;
    //    int boxY = preferredY >= 0 ? preferredY : Math.Min(Math.Max(0, anchorY + 2), Math.Max(0, imageH - boxH));

    //    // Semi-transparent black background.
    //    FillRect(pixels, imageW, imageH, boxX, boxY, boxW, boxH, 0, 0, 0, 180);

    //    // Draw glyphs.
    //    int cursorX = boxX + 2;
    //    foreach (var (buf, gw, gh) in glyphs)
    //    {
    //        BlendGlyph(pixels, imageW, imageH, cursorX, boxY + 2, buf, gw, gh, r, g, b);
    //        cursorX += gw + 1;
    //    }
    //}

    // Pixel primitives.

    private static void SetPixel(byte[] pixels, int imageW, int imageH, int x, int y, byte r, byte g, byte b, byte a)
    {
        if ((uint)x >= imageW || (uint)y >= imageH)
            return;

        int idx = (y * imageW + x) * 4;
        pixels[idx] = r;
        pixels[idx + 1] = g;
        pixels[idx + 2] = b;
        pixels[idx + 3] = a;
    }

    private static void FillRect(byte[] pixels, int imageW, int imageH, int x, int y, int w, int h, byte r, byte g, byte b, byte alpha)
    {
        int startX = Math.Max(x, 0);
        int startY = Math.Max(y, 0);
        int endX = Math.Min(x + w, imageW);
        int endY = Math.Min(y + h, imageH);

        for (int py = startY; py < endY; py++)
        {
            for (int px = startX; px < endX; px++)
            {
                int idx = (py * imageW + px) * 4;
                float a = alpha / 255f;
                float inv = 1f - a;
                pixels[idx] = (byte)Math.Clamp((int)Math.Round(r * a + pixels[idx] * inv), 0, 255);
                pixels[idx + 1] = (byte)Math.Clamp((int)Math.Round(g * a + pixels[idx + 1] * inv), 0, 255);
                pixels[idx + 2] = (byte)Math.Clamp((int)Math.Round(b * a + pixels[idx + 2] * inv), 0, 255);
                pixels[idx + 3] = 255;
            }
        }
    }

    private static void BlendGlyph(byte[] pixels, int imageW, int imageH, int x, int y, byte[] glyphBuf, int gw, int gh, byte r, byte g, byte b)
    {
        for (int gy = 0; gy < gh; gy++)
        {
            for (int gx = 0; gx < gw; gx++)
            {
                int gIdx = (gy * gw + gx) * 4;
                byte alpha = glyphBuf[gIdx + 3];
                if (alpha == 0) continue;

                int px = x + gx;
                int py = y + gy;
                if ((uint)px >= imageW || (uint)py >= imageH) continue;

                int idx = (py * imageW + px) * 4;
                float a = alpha / 255f;
                float inv = 1f - a;
                pixels[idx] = (byte)Math.Clamp((int)Math.Round(r * a + pixels[idx] * inv), 0, 255);
                pixels[idx + 1] = (byte)Math.Clamp((int)Math.Round(g * a + pixels[idx + 1] * inv), 0, 255);
                pixels[idx + 2] = (byte)Math.Clamp((int)Math.Round(b * a + pixels[idx + 2] * inv), 0, 255);
                pixels[idx + 3] = 255;
            }
        }
    }

    // Internal types.

    public class Prediction
    {
        public Box Box { get; set; }
        public string Label { get; set; }
        public float Confidence { get; set; }
    }

    public class Box
    {
        public float Xmin { get; set; }
        public float Ymin { get; set; }
        public float Xmax { get; set; }
        public float Ymax { get; set; }

        public Box(float xmin, float ymin, float xmax, float ymax)
        {
            Xmin = xmin;
            Ymin = ymin;
            Xmax = xmax;
            Ymax = ymax;
        }
    }

    public sealed class FasterRcnnResult
    {
        public List<Prediction> Predictions { get; set; } = new();

        public byte[] AnnotatedImage { get; set; } = [];
    }
}
