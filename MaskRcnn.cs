// Copyright (c) SeasonEngine and contributors.
// Licensed under the MIT License.
// https://github.com/SeasonRealms/SeasonVision

namespace Season.Vision;

public class MaskRcnn
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

    public class Prediction
    {
        public Box Box { get; set; } = null!;
        public string Label { get; set; } = "";
        public float Confidence { get; set; }
        public float[] Mask { get; set; } = null!;
        public bool MaskIsProbability { get; set; }

        public string Summary
        {
            get
            {
                string mask = Mask?.Length > 0 ? Mask.Length.ToString() : "0"; // String.Join(", ", Mask) : String.Empty;
                return $"Xmin:{Box.Xmin} Ymin:{Box.Ymin} Xmax:{Box.Xmax} Ymax:{Box.Ymax} Label:{Label} Mask:{mask} Confidence:{Confidence} MaskIsProbability:{MaskIsProbability}";
            }
        }
    }

    public class Box
    {
        public float Xmin { get; set; }
        public float Ymin { get; set; }
        public float Xmax { get; set; }
        public float Ymax { get; set; }

        public Box(float xmin, float ymin, float xmax, float ymax)
        {
            Xmin = xmin; Ymin = ymin; Xmax = xmax; Ymax = ymax;
        }
    }

    public sealed class MaskRcnnResult
    {
        public List<Prediction> Predictions { get; set; } = new();

        public byte[] AnnotatedImage { get; set; } = [];

        public string Summary
        {
            get
            {
                var pres = Predictions.Select(pre => pre.Summary);

                return String.Join("\r\n", pres);
            }
        }
    }


    private static readonly (byte R, byte G, byte B)[] ClassPalette =
    {
        (255, 99, 71),   (50, 205, 50),   (30, 144, 255), (255, 215, 0),
        (186, 85, 211),  (0, 206, 209),   (255, 140, 0),  (220, 20, 60),
        (100, 149, 237), (60, 179, 113),  (238, 130, 238), (255, 255, 0),
        (70, 130, 180),  (244, 164, 96),  (218, 112, 214), (255, 192, 203),
    };

    public static MaskRcnnResult Detect(InferenceSession session, ReadOnlySpan<byte> imageData, int width, int height, bool createAnnotatedImage = false)
    {
        var result = new MaskRcnnResult();

        const int shortSide = 800;
        const int alignment = 32;
        const float minConfidence = 0.5f;
        const int maskSize = 28;
        const int fontSize = 14;

        //var modelFilePath = DeviceServices.Core.LoadFilePath(model);

        var rgb = ImageProcessor.ExtractRgb(imageData, width, height);
        var mean = new[] { 102.9801f, 115.9465f, 122.7717f };
        float[] chw = ImageProcessor.ResizePadBgrNormalize(
            rgb, width, height, shortSide, alignment, mean);

        float scale = (float)shortSide / Math.Min(width, height);
        int newW = Math.Max(1, (int)(width * scale));
        int newH = Math.Max(1, (int)(height * scale));
        int paddedW = ((newW + alignment - 1) / alignment) * alignment;
        int paddedH = ((newH + alignment - 1) / alignment) * alignment;

        var input = new DenseTensor<float>(chw, new[] { 3, paddedH, paddedW });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("image", input)
        };

        using var results = session.Run(inputs);
        var resultsArray = results.ToArray();

        float[] boxes = resultsArray[0].AsEnumerable<float>().ToArray();
        long[] labels = resultsArray[1].AsEnumerable<long>().ToArray();
        float[] confidences = resultsArray[2].AsEnumerable<float>().ToArray();
        float[] masks = resultsArray[3].AsEnumerable<float>().ToArray();

        var maskTensor = resultsArray[3].AsTensor<float>();
        int[] maskDims = maskTensor.Dimensions.ToArray();
        bool maskLooksLikeProbabilities = masks.Length > 0 && masks.All(v => v >= 0f && v <= 1f);

        float invScale = 1f / scale;
        int numDetections = confidences.Length;
        var predictions = new List<Prediction>();

        Debug.WriteLine($"[MaskRcnn] detections={numDetections} boxes={boxes.Length / 4} masks={masks.Length} dims=[{string.Join(",", maskDims)}] expectedMaskPlane={maskSize * maskSize} valuesAreProbabilities={maskLooksLikeProbabilities}");

        for (int i = 0; i + 3 < boxes.Length; i += 4)
        {
            int idx = i / 4;
            if (idx >= numDetections || confidences[idx] < minConfidence)
                continue;

            int labelIdx = (int)labels[idx];
            string label = labelIdx > 0 && labelIdx < Labels.Length
                ? Labels[labelIdx]
                : $"cls_{labelIdx}";

            int maskPlane = maskSize * maskSize;
            float[] mask28 = new float[maskPlane];
            int maskBase = idx * maskPlane;
            for (int m = 0; m < maskPlane; m++)
                mask28[m] = masks[maskBase + m];

            var prediction = new Prediction
            {
                Box = new Box(
                    boxes[i] * invScale,
                    boxes[i + 1] * invScale,
                    boxes[i + 2] * invScale,
                    boxes[i + 3] * invScale),
                Label = label,
                Confidence = confidences[idx],
                Mask = mask28,
                MaskIsProbability = maskLooksLikeProbabilities
            };

            predictions.Add(prediction);
            LogMaskDebug(idx, prediction, width, height);
        }

        result.Predictions = predictions;

        if (createAnnotatedImage)
        {
            int pixelCount = width * height;
            var srcData = imageData;
            var pixels = new byte[pixelCount * 4];
            for (int pi = 0; pi < pixelCount; pi++)
            {
                int s = pi * 4;
                int d = pi * 4;
                pixels[d] = srcData[s];
                pixels[d + 1] = srcData[s + 1];
                pixels[d + 2] = srcData[s + 2];
                pixels[d + 3] = srcData[s + 3];
            }

            if (predictions.Count > 0)
            {
                int colorIdx = 0;

                foreach (var p in predictions)
                {
                    var color = ClassPalette[colorIdx % ClassPalette.Length];
                    colorIdx++;

                    PaintMask(pixels, width, height, p, color.R, color.G, color.B);

                    //DrawLabel(pixels, imageResult.Width, imageResult.Height,
                    //    p, font, fontSize, color.R, color.G, color.B);
                }
            }

            result.AnnotatedImage = pixels;
        }

        return result;
    }

    private static void PaintMask(
        byte[] pixels, int imageW, int imageH,
        Prediction p, byte r, byte g, byte b)
    {
        const int maskSize = 28;
        const float fillAlpha = 0.72f;
        const float maskThreshold = 0.5f;
        const float smoothBand = 0.12f;

        int bx1 = Math.Max(0, (int)MathF.Floor(p.Box.Xmin));
        int by1 = Math.Max(0, (int)MathF.Floor(p.Box.Ymin));
        int bx2 = Math.Min(imageW, (int)MathF.Ceiling(p.Box.Xmax));
        int by2 = Math.Min(imageH, (int)MathF.Ceiling(p.Box.Ymax));

        int boxW = bx2 - bx1;
        int boxH = by2 - by1;
        if (boxW <= 0 || boxH <= 0) return;

        float[] probMask = new float[maskSize * maskSize];
        int srcActive = 0;
        int srcStrong = 0;
        for (int i = 0; i < probMask.Length; i++)
        {
            float prob = ToMaskProbability(p.Mask[i], p.MaskIsProbability);
            probMask[i] = prob;
            if (prob >= maskThreshold)
                srcActive++;
            if (prob >= 0.8f)
                srcStrong++;
        }

        int paintedPixels = 0;
        int strongPixels = 0;
        float maxProb = 0f;
        float minProb = 1f;

        for (int py = by1; py < by2; py++)
        {
            float v = (py + 0.5f - p.Box.Ymin) / Math.Max(1f, p.Box.Ymax - p.Box.Ymin);
            for (int px = bx1; px < bx2; px++)
            {
                float u = (px + 0.5f - p.Box.Xmin) / Math.Max(1f, p.Box.Xmax - p.Box.Xmin);
                float prob = SampleMaskProbability(probMask, maskSize, u, v);
                maxProb = Math.Max(maxProb, prob);
                minProb = Math.Min(minProb, prob);

                float coverage = SmoothStep(maskThreshold - smoothBand, maskThreshold + smoothBand, prob);
                if (coverage <= 0f)
                    continue;

                paintedPixels++;
                if (prob >= 0.8f)
                    strongPixels++;

                BlendPixel(pixels, imageW, imageH, px, py, r, g, b, fillAlpha * coverage);
            }
        }

        int boxPixels = Math.Max(1, boxW * boxH);
        Debug.WriteLine(
            $"[MaskRcnnPaint] label={p.Label} conf={p.Confidence:0.00} box=({bx1},{by1})-({bx2},{by2}) boxPixels={boxPixels} " +
            $"srcActive={srcActive}/{maskSize * maskSize} srcStrong={srcStrong} painted={paintedPixels} paintedRatio={(float)paintedPixels / boxPixels:0.000} strongPainted={strongPixels} " +
            $"sampleProbRange={minProb:0.000}-{maxProb:0.000} valuesAreProbabilities={p.MaskIsProbability}");
    }

    private static void LogMaskDebug(int idx, Prediction p, int imageW, int imageH)
    {
        const int maskSize = 28;
        const float threshold = 0.5f;

        float minValue = float.MaxValue;
        float maxValue = float.MinValue;
        float minProb = 1f;
        float maxProb = 0f;
        float sumProb = 0f;
        int active = 0;
        int strong = 0;
        int[] rows = new int[maskSize];
        int[] cols = new int[maskSize];

        for (int y = 0; y < maskSize; y++)
        {
            for (int x = 0; x < maskSize; x++)
            {
                int i = y * maskSize + x;
                float value = p.Mask[i];
                float prob = ToMaskProbability(value, p.MaskIsProbability);
                minValue = Math.Min(minValue, value);
                maxValue = Math.Max(maxValue, value);
                minProb = Math.Min(minProb, prob);
                maxProb = Math.Max(maxProb, prob);
                sumProb += prob;
                if (prob >= threshold)
                {
                    active++;
                    rows[y]++;
                    cols[x]++;
                }
                if (prob >= 0.8f)
                    strong++;
            }
        }

        int bx1 = Math.Max(0, (int)MathF.Floor(p.Box.Xmin));
        int by1 = Math.Max(0, (int)MathF.Floor(p.Box.Ymin));
        int bx2 = Math.Min(imageW, (int)MathF.Ceiling(p.Box.Xmax));
        int by2 = Math.Min(imageH, (int)MathF.Ceiling(p.Box.Ymax));
        int boxW = Math.Max(0, bx2 - bx1);
        int boxH = Math.Max(0, by2 - by1);

        int rowMax = rows.Max();
        int colMax = cols.Max();
        string rowSummary = string.Join(",", rows.Select(v => v.ToString()));
        string colSummary = string.Join(",", cols.Select(v => v.ToString()));
        string asciiMask = BuildMaskAscii(p.Mask, maskSize, threshold, p.MaskIsProbability);

        Debug.WriteLine(
            $"[MaskRcnnDebug] idx={idx} label={p.Label} conf={p.Confidence:0.00} box=({p.Box.Xmin:0.0},{p.Box.Ymin:0.0},{p.Box.Xmax:0.0},{p.Box.Ymax:0.0}) " +
            $"boxSize={boxW}x{boxH} maskValueRange={minValue:0.000}..{maxValue:0.000} probRange={minProb:0.000}..{maxProb:0.000} avgProb={sumProb / (maskSize * maskSize):0.000} " +
            $"active={active}/{maskSize * maskSize} strong={strong} rowMax={rowMax} colMax={colMax} valuesAreProbabilities={p.MaskIsProbability}");
        Debug.WriteLine($"[MaskRcnnRows] idx={idx} rows={rowSummary}");
        Debug.WriteLine($"[MaskRcnnCols] idx={idx} cols={colSummary}");
        Debug.WriteLine($"[MaskRcnnAscii] idx={idx}\n{asciiMask}");
    }

    private static string BuildMaskAscii(float[] maskValues, int maskSize, float threshold, bool maskIsProbability)
    {
        var lines = new List<string>(maskSize);
        for (int y = 0; y < maskSize; y++)
        {
            char[] row = new char[maskSize];
            for (int x = 0; x < maskSize; x++)
            {
                float prob = ToMaskProbability(maskValues[y * maskSize + x], maskIsProbability);
                row[x] = prob >= 0.8f ? '#' : prob >= threshold ? '+' : '.';
            }
            lines.Add(new string(row));
        }
        return string.Join(Environment.NewLine, lines);
    }

    private static float ToMaskProbability(float value, bool maskIsProbability)
        => maskIsProbability ? value : Sigmoid(value);

    private static void DrawBoxOutline(
        byte[] pixels, int w, int h,
        Prediction p, byte r, byte g, byte b, int thickness)
    {
        int xmin = (int)p.Box.Xmin;
        int ymin = (int)p.Box.Ymin;
        int xmax = (int)p.Box.Xmax;
        int ymax = (int)p.Box.Ymax;

        for (int t = 0; t < thickness; t++)
        {
            for (int dx = xmin; dx <= xmax; dx++)
            {
                SetPixel(pixels, w, h, dx, ymin + t, r, g, b, 255);
                SetPixel(pixels, w, h, dx, ymax - t, r, g, b, 255);
            }
            for (int dy = ymin; dy <= ymax; dy++)
            {
                SetPixel(pixels, w, h, xmin + t, dy, r, g, b, 255);
                SetPixel(pixels, w, h, xmax - t, dy, r, g, b, 255);
            }
        }
    }

    //private static void DrawLabel(
    //    byte[] pixels, int imageW, int imageH,
    //    Prediction p, Season.Fonts.Font font, int fontSize, byte r, byte g, byte b)
    //{
    //    var text = $"{p.Label} {p.Confidence:0.00}";
    //    if (string.IsNullOrWhiteSpace(text)) return;

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

    //    if (glyphs.Count == 0) return;

    //    int textW = glyphs.Sum(g => g.W + 1);
    //    int boxW = textW + 4;
    //    int boxH = textH + 4;

    //    int anchorX = Math.Max(2, (int)p.Box.Xmin);
    //    int anchorY = (int)p.Box.Ymin;
    //    int boxX = Math.Clamp(anchorX, 0, Math.Max(0, imageW - boxW));
    //    int preferredY = anchorY - boxH - 2;
    //    int boxY = preferredY >= 0 ? preferredY : Math.Min(Math.Max(0, anchorY + 2), Math.Max(0, imageH - boxH));

    //    FillRect(pixels, imageW, imageH, boxX, boxY, boxW, boxH, 0, 0, 0, 160);

    //    int cursorX = boxX + 2;
    //    foreach (var (buf, gw, gh) in glyphs)
    //    {
    //        BlendGlyph(pixels, imageW, imageH, cursorX, boxY + 2, buf, gw, gh, r, g, b);
    //        cursorX += gw + 1;
    //    }
    //}

    private static void SetPixel(byte[] pixels, int imageW, int imageH,
        int x, int y, byte r, byte g, byte b, byte a)
    {
        if ((uint)x >= imageW || (uint)y >= imageH) return;
        int idx = (y * imageW + x) * 4;
        pixels[idx] = r;
        pixels[idx + 1] = g;
        pixels[idx + 2] = b;
        pixels[idx + 3] = a;
    }

    private static void FillRect(byte[] pixels, int imageW, int imageH,
        int x, int y, int w, int h, byte r, byte g, byte b, byte alpha)
    {
        int sx = Math.Max(x, 0);
        int sy = Math.Max(y, 0);
        int ex = Math.Min(x + w, imageW);
        int ey = Math.Min(y + h, imageH);

        for (int py = sy; py < ey; py++)
        {
            for (int px = sx; px < ex; px++)
            {
                BlendPixel(pixels, imageW, imageH, px, py, r, g, b, alpha / 255f);
                int idx = (py * imageW + px) * 4;
                pixels[idx + 3] = 255;
            }
        }
    }

    private static void BlendGlyph(byte[] pixels, int imageW, int imageH,
        int x, int y, byte[] glyphBuf, int gw, int gh, byte r, byte g, byte b)
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

                BlendPixel(pixels, imageW, imageH, px, py, r, g, b, alpha / 255f);
                int idx = (py * imageW + px) * 4;
                pixels[idx + 3] = 255;
            }
        }
    }

    private static void BlendPixel(byte[] pixels, int imageW, int imageH,
        int x, int y, byte r, byte g, byte b, float alpha)
    {
        if ((uint)x >= imageW || (uint)y >= imageH)
            return;

        int idx = (y * imageW + x) * 4;
        float a = Math.Clamp(alpha, 0f, 1f);
        float inv = 1f - a;
        pixels[idx] = (byte)Math.Clamp((int)Math.Round(r * a + pixels[idx] * inv), 0, 255);
        pixels[idx + 1] = (byte)Math.Clamp((int)Math.Round(g * a + pixels[idx + 1] * inv), 0, 255);
        pixels[idx + 2] = (byte)Math.Clamp((int)Math.Round(b * a + pixels[idx + 2] * inv), 0, 255);
        pixels[idx + 3] = 255;
    }

    private static float SampleBilinear(float[] data, int width, int height, float x, float y)
    {
        float cx = Math.Clamp(x, 0f, width - 1);
        float cy = Math.Clamp(y, 0f, height - 1);

        int x0 = (int)cx;
        int y0 = (int)cy;
        int x1 = Math.Min(x0 + 1, width - 1);
        int y1 = Math.Min(y0 + 1, height - 1);
        float fx = cx - x0;
        float fy = cy - y0;

        float v00 = data[y0 * width + x0];
        float v10 = data[y0 * width + x1];
        float v01 = data[y1 * width + x0];
        float v11 = data[y1 * width + x1];
        float top = (1f - fx) * v00 + fx * v10;
        float bottom = (1f - fx) * v01 + fx * v11;
        return (1f - fy) * top + fy * bottom;
    }

    private static float SampleMaskProbability(float[] data, int maskSize, float u, float v)
    {
        float x = Math.Clamp(u * maskSize - 0.5f, 0f, maskSize - 1f);
        float y = Math.Clamp(v * maskSize - 0.5f, 0f, maskSize - 1f);
        return SampleBilinear(data, maskSize, maskSize, x, y);
    }

    private static float SmoothStep(float edge0, float edge1, float x)
    {
        if (edge0 == edge1)
            return x < edge0 ? 0f : 1f;

        float t = Math.Clamp((x - edge0) / (edge1 - edge0), 0f, 1f);
        return t * t * (3f - 2f * t);
    }

    private static float Sigmoid(float x) => 1.0f / (1.0f + MathF.Exp(-x));
}
