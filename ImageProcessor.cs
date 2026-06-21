// Copyright (c) SeasonEngine and contributors.
// Licensed under the MIT License.
// https://github.com/SeasonRealms/SeasonVision

namespace SeasonVision;

/// <summary>
/// Shared image preprocessing helpers for vision models.
/// Provides RGBA-to-RGB extraction, bilinear resizing, letterbox and
/// center-crop transforms, and normalization helpers.
/// </summary>
internal static class ImageProcessor
{
    /// <summary>
    /// Extracts a three-channel RGB byte array from an RGBA image buffer.
    /// The source buffer is assumed to be RGBA8, so the alpha channel is dropped.
    /// </summary>
    public static byte[] ExtractRgb(ReadOnlySpan<byte> imageData, int width, int height)
    {
        int pixelCount = width * height;
        var data = imageData;

        var result = new byte[pixelCount * 3];
        for (int i = 0; i < pixelCount; i++)
        {
            int src = i * 4;
            int dst = i * 3;
            result[dst] = data[src];         // R
            result[dst + 1] = data[src + 1]; // G
            result[dst + 2] = data[src + 2]; // B
        }
        return result;
    }

    /// <summary>
    /// Normalizes the input image into an RGBA byte array.
    /// The source buffer is assumed to be RGBA8, so the bytes are copied directly.
    /// </summary>
    public static byte[] EnsureRgba(ReadOnlySpan<byte> imageData, int width, int height)
    {
        int pixelCount = width * height;
        var data = imageData;

        var rgba = new byte[pixelCount * 4];
        for (int i = 0; i < pixelCount * 4; i++)
            rgba[i] = data[i];
        return rgba;
    }

    /// <summary>
    /// Resizes directly to the target size and normalizes the result into an
    /// NCHW float array using the provided mean and standard deviation.
    /// Intended for models such as Ultraface that use a fixed input size.
    /// </summary>
    public static float[] ResizeNormalizeToNchw(
        byte[] srcRgb, int srcW, int srcH, int dstW, int dstH,
        float[] mean, float[] stddev, bool scaleToUnitInterval)
    {
        ValidateChannels(mean, stddev);

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

                for (int c = 0; c < 3; c++)
                {
                    float val = SampleBilinearChannel(srcRgb, srcW, srcH, sx, sy, c);
                    if (scaleToUnitInterval)
                        val /= 255.0f;

                    result[c * dstW * dstH + dstIdx] = (val - mean[c]) / stddev[c];
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Resizes an image with bilinear interpolation and returns an RGBA buffer.
    /// </summary>
    public static byte[] Resize(ReadOnlySpan<byte> imageData, int width, int height, int dstW, int dstH)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dstW);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dstH);

        var rgb = ExtractRgb(imageData, width, height);
        var resized = ResizeRgb(rgb, width, height, dstW, dstH);

        // Convert RGB back to RGBA.
        int pixelCount = dstW * dstH;
        var rgba = new byte[pixelCount * 4];
        for (int i = 0; i < pixelCount; i++)
        {
            int s = i * 3;
            int d = i * 4;
            rgba[d] = resized[s];
            rgba[d + 1] = resized[s + 1];
            rgba[d + 2] = resized[s + 2];
            rgba[d + 3] = 255;
        }

        return rgba;
    }

    /// <summary>
    /// Resizes an image while preserving the aspect ratio so the shorter edge
    /// matches the requested length.
    /// </summary>
    public static byte[] ResizeShortestEdge(ReadOnlySpan<byte> imageData, int width, int height, int shortestEdge)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(shortestEdge);

        float ratio = (float)shortestEdge / Math.Min(width, height);
        int dstW = Math.Max(1, (int)MathF.Ceiling(width * ratio));
        int dstH = Math.Max(1, (int)MathF.Ceiling(height * ratio));
        return Resize(imageData, width, height, dstW, dstH);
    }

    /// <summary>
    /// Crops a rectangular region from an RGB image and returns RGB data.
    /// </summary>
    public static byte[] CropRgb(byte[] srcRgb, int srcW, int srcH, int x, int y, int width, int height)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(x);
        ArgumentOutOfRangeException.ThrowIfNegative(y);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);

        if (x + width > srcW || y + height > srcH)
            throw new ArgumentOutOfRangeException(nameof(width), "The crop region exceeds the source image bounds.");

        var result = new byte[width * height * 3];

        for (int row = 0; row < height; row++)
        {
            int srcOffset = ((y + row) * srcW + x) * 3;
            int dstOffset = row * width * 3;
            Array.Copy(srcRgb, srcOffset, result, dstOffset, width * 3);
        }

        return result;
    }

    /// <summary>
    /// Resizes an RGB image with bilinear interpolation.
    /// Input: byte[] in HWC layout. Output: float[] in NCHW layout normalized by /255.
    /// </summary>
    public static float[] BilinearResize(byte[] srcRgb, int srcW, int srcH, int dstW, int dstH)
    {
        float scaleX = (float)srcW / dstW;
        float scaleY = (float)srcH / dstH;
        float[] result = new float[3 * dstW * dstH];

        for (int dy = 0; dy < dstH; dy++)
        {
            float sy = dy * scaleY;
            int y0 = (int)sy;
            int y1 = Math.Min(y0 + 1, srcH - 1);
            float fy = sy - y0;

            for (int dx = 0; dx < dstW; dx++)
            {
                float sx = dx * scaleX;
                int x0 = (int)sx;
                int x1 = Math.Min(x0 + 1, srcW - 1);
                float fx = sx - x0;

                int dstIdx = dy * dstW + dx;

                for (int c = 0; c < 3; c++)
                {
                    int srcIdx00 = (y0 * srcW + x0) * 3 + c;
                    int srcIdx10 = (y0 * srcW + x1) * 3 + c;
                    int srcIdx01 = (y1 * srcW + x0) * 3 + c;
                    int srcIdx11 = (y1 * srcW + x1) * 3 + c;

                    float v00 = srcRgb[srcIdx00];
                    float v10 = srcRgb[srcIdx10];
                    float v01 = srcRgb[srcIdx01];
                    float v11 = srcRgb[srcIdx11];

                    float top = (1 - fx) * v00 + fx * v10;
                    float bot = (1 - fx) * v01 + fx * v11;
                    float val = ((1 - fy) * top + fy * bot) / 255.0f;

                    result[c * dstW * dstH + dstIdx] = val;
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Resizes the shorter edge to the requested size, applies a center crop,
    /// and normalizes the result into an NCHW float array.
    /// Intended for the 256 -> 224 preprocessing flow used by the MobileNet v2 example.
    /// </summary>
    public static float[] ResizeShortestEdgeCenterCropNormalize(
        byte[] srcRgb, int srcW, int srcH,
        int resizeShortestEdge, int cropW, int cropH,
        float[] mean, float[] stddev)
    {
        ValidateChannels(mean, stddev);

        float scale = (float)resizeShortestEdge / Math.Min(srcW, srcH);
        float resizedW = MathF.Ceiling(srcW * scale);
        float resizedH = MathF.Ceiling(srcH * scale);
        float cropX = Math.Max(0f, (resizedW - cropW) / 2f);
        float cropY = Math.Max(0f, (resizedH - cropH) / 2f);
        float invScale = 1f / scale;
        float[] result = new float[3 * cropW * cropH];

        for (int dy = 0; dy < cropH; dy++)
        {
            float sy = (cropY + dy) * invScale;
            for (int dx = 0; dx < cropW; dx++)
            {
                float sx = (cropX + dx) * invScale;
                int dstIdx = dy * cropW + dx;

                for (int c = 0; c < 3; c++)
                {
                    float val = SampleBilinearChannel(srcRgb, srcW, srcH, sx, sy, c) / 255.0f;
                    result[c * cropW * cropH + dstIdx] = (val - mean[c]) / stddev[c];
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Letterbox resize plus normalization in a YOLO-style pipeline.
    /// Preserves the aspect ratio while resizing to the target dimensions and
    /// fills the remaining area with gray (114/255).
    /// Returns data in NCHW float[] format.
    /// </summary>
    public static float[] LetterboxResizeNormalize(
        byte[] srcRgb, int srcW, int srcH, int dstW, int dstH)
    {
        // Compute the resize scale while preserving the aspect ratio.
        float scale = Math.Min((float)dstW / srcW, (float)dstH / srcH);
        int newW = Math.Max(1, (int)(srcW * scale));
        int newH = Math.Max(1, (int)(srcH * scale));

        // Resize with bilinear interpolation.
        float[] resized = BilinearResize(srcRgb, srcW, srcH, newW, newH);

        // Fill the padding area with gray (114/255 ~= 0.447).
        const float gray = 114.0f / 255.0f;
        float[] result = new float[3 * dstW * dstH];
        for (int i = 0; i < result.Length; i++)
            result[i] = gray;

        int padX = (dstW - newW) / 2;
        int padY = (dstH - newH) / 2;

        // Copy the resized image into the center of the destination buffer.
        for (int c = 0; c < 3; c++)
        {
            int chOffset = c * dstW * dstH;
            int srcChOffset = c * newW * newH;

            for (int y = 0; y < newH; y++)
            {
                int dstRow = (y + padY) * dstW + padX;
                int srcRow = y * newW;

                for (int x = 0; x < newW; x++)
                {
                    result[chOffset + dstRow + x] = resized[srcChOffset + srcRow + x];
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Center-crop resize plus ImageNet mean and standard deviation
    /// normalization in a ResNet-style pipeline.
    /// First rescales the image so the shorter edge matches the target, then
    /// performs a centered square crop.
    /// Returns data in NCHW float[] format.
    /// </summary>
    public static float[] CenterCropResizeNormalize(
        byte[] srcRgb, int srcW, int srcH, int dstW, int dstH,
        float[] mean, float[] stddev)
    {
        // Compute the scale so the shorter edge matches the target and the longer edge can be cropped.
        float scale = Math.Max((float)dstW / srcW, (float)dstH / srcH);
        float intW = srcW * scale;
        float intH = srcH * scale;
        float cropX = (intW - dstW) / 2f;
        float cropY = (intH - dstH) / 2f;

        float invScale = 1f / scale;
        float[] result = new float[3 * dstW * dstH];

        for (int dy = 0; dy < dstH; dy++)
        {
            float sy = (cropY + dy) * invScale;
            int y0 = (int)sy;
            int y1 = Math.Min(y0 + 1, srcH - 1);
            float fy = sy - y0;

            for (int dx = 0; dx < dstW; dx++)
            {
                float sx = (cropX + dx) * invScale;
                int x0 = (int)sx;
                int x1 = Math.Min(x0 + 1, srcW - 1);
                float fx = sx - x0;

                int dstIdx = dy * dstW + dx;

                for (int c = 0; c < 3; c++)
                {
                    int srcIdx00 = (y0 * srcW + x0) * 3 + c;
                    int srcIdx10 = (y0 * srcW + x1) * 3 + c;
                    int srcIdx01 = (y1 * srcW + x0) * 3 + c;
                    int srcIdx11 = (y1 * srcW + x1) * 3 + c;

                    float v00 = srcRgb[srcIdx00];
                    float v10 = srcRgb[srcIdx10];
                    float v01 = srcRgb[srcIdx01];
                    float v11 = srcRgb[srcIdx11];

                    float top = (1 - fx) * v00 + fx * v10;
                    float bot = (1 - fx) * v01 + fx * v11;
                    float val = ((1 - fy) * top + fy * bot) / 255.0f;

                    result[c * dstW * dstH + dstIdx] = (val - mean[c]) / stddev[c];
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Catmull-Rom cubic kernel matching ImageSharp KnownResamplers.Bicubic.
    /// </summary>
    private static float Cubic(float x)
    {
        const float a = -0.5f;
        x = Math.Abs(x);
        if (x > 2.0f) return 0;
        float x2 = x * x;
        if (x <= 1.0f) return (a + 2) * x * x2 - (a + 3) * x2 + 1;
        return a * x * x2 - 5 * a * x2 + 8 * a * x - 4 * a;
    }

    /// <summary>
    /// Aspect-ratio resize to the target short edge, 32-aligned padding, BGR
    /// channel reordering, and mean subtraction for a Faster R-CNN-style pipeline.
    /// Resizing uses Catmull-Rom bicubic interpolation to match the torchvision
    /// and ImageSharp preprocessing behavior.
    /// Input: byte[] HWC RGB. Output: float[] CHW BGR ([B, G, R]) without /255
    /// scaling or standard deviation normalization.
    /// Padding values are zero and are applied on the right and bottom sides.
    /// </summary>
    /// <param name="srcRgb">RGB byte array with three channels.</param>
    /// <param name="srcW">Source image width.</param>
    /// <param name="srcH">Source image height.</param>
    /// <param name="shortSide">Target size for the shorter edge, typically 800 for Faster R-CNN.</param>
    /// <param name="alignment">Padding alignment, typically 32.</param>
    /// <param name="mean">Mean values for the BGR channels, for example [102.9801f, 115.9465f, 122.7717f].</param>
    /// <returns>A float array shaped as CHW BGR data with length 3 * paddedH * paddedW.</returns>
    public static float[] ResizePadBgrNormalize(
        byte[] srcRgb, int srcW, int srcH,
        int shortSide, int alignment,
        float[] mean)
    {
        // Resize proportionally so the shorter edge equals shortSide.
        float scale = (float)shortSide / Math.Min(srcW, srcH);
        int newW = Math.Max(1, (int)(srcW * scale));
        int newH = Math.Max(1, (int)(srcH * scale));

        // Pad to the requested alignment using right and bottom padding.
        int paddedW = ((newW + alignment - 1) / alignment) * alignment;
        int paddedH = ((newH + alignment - 1) / alignment) * alignment;

        float[] result = new float[3 * paddedW * paddedH]; // Defaults to zeros.
        float invScale = 1f / scale;

        int chPlane = paddedW * paddedH;
        int chB = 0;             // B channel offset.
        int chG = chPlane;       // G channel offset.
        int chR = 2 * chPlane;   // R channel offset.

        // Local clamp helpers; the closure overhead is negligible here.
        int ClampY(int v) => v < 0 ? 0 : (v >= srcH ? srcH - 1 : v);
        int ClampX(int v) => v < 0 ? 0 : (v >= srcW ? srcW - 1 : v);

        for (int dy = 0; dy < newH; dy++)
        {
            float sy = dy * invScale;
            int syInt = (int)sy;
            int syBase = syInt - 1;

            // Four vertical weights.
            float wy0 = Cubic(sy - syBase);
            float wy1 = Cubic(sy - (syBase + 1));
            float wy2 = Cubic(sy - (syBase + 2));
            float wy3 = Cubic(sy - (syBase + 3));

            int sy0 = ClampY(syBase);
            int sy1 = ClampY(syBase + 1);
            int sy2 = ClampY(syBase + 2);
            int sy3 = ClampY(syBase + 3);

            int dstRow = dy * paddedW;
            int row0 = sy0 * srcW * 3;
            int row1 = sy1 * srcW * 3;
            int row2 = sy2 * srcW * 3;
            int row3 = sy3 * srcW * 3;

            for (int dx = 0; dx < newW; dx++)
            {
                float sx = dx * invScale;
                int sxInt = (int)sx;
                int sxBase = sxInt - 1;

                // Four horizontal weights.
                float wx0 = Cubic(sx - sxBase);
                float wx1 = Cubic(sx - (sxBase + 1));
                float wx2 = Cubic(sx - (sxBase + 2));
                float wx3 = Cubic(sx - (sxBase + 3));

                int sx0 = ClampX(sxBase);
                int sx1 = ClampX(sxBase + 1);
                int sx2 = ClampX(sxBase + 2);
                int sx3 = ClampX(sxBase + 3);

                int dstIdx = dstRow + dx;
                int x0_3 = sx0 * 3, x1_3 = sx1 * 3, x2_3 = sx2 * 3, x3_3 = sx3 * 3;

                // Apply a 4x4 bicubic convolution independently to each channel.
                for (int c = 0; c < 3; c++)
                {
                    float v00 = srcRgb[row0 + x0_3 + c], v01 = srcRgb[row0 + x1_3 + c];
                    float v02 = srcRgb[row0 + x2_3 + c], v03 = srcRgb[row0 + x3_3 + c];
                    float v10 = srcRgb[row1 + x0_3 + c], v11 = srcRgb[row1 + x1_3 + c];
                    float v12 = srcRgb[row1 + x2_3 + c], v13 = srcRgb[row1 + x3_3 + c];
                    float v20 = srcRgb[row2 + x0_3 + c], v21 = srcRgb[row2 + x1_3 + c];
                    float v22 = srcRgb[row2 + x2_3 + c], v23 = srcRgb[row2 + x3_3 + c];
                    float v30 = srcRgb[row3 + x0_3 + c], v31 = srcRgb[row3 + x1_3 + c];
                    float v32 = srcRgb[row3 + x2_3 + c], v33 = srcRgb[row3 + x3_3 + c];

                    float val =
                        (v00 * wx0 + v01 * wx1 + v02 * wx2 + v03 * wx3) * wy0 +
                        (v10 * wx0 + v11 * wx1 + v12 * wx2 + v13 * wx3) * wy1 +
                        (v20 * wx0 + v21 * wx1 + v22 * wx2 + v23 * wx3) * wy2 +
                        (v30 * wx0 + v31 * wx1 + v32 * wx2 + v33 * wx3) * wy3;

                    // BGR channel mapping: source c=0 (R) -> chR, c=1 (G) -> chG, c=2 (B) -> chB.
                    // Mean mapping: c=0 -> mean[2] (R mean), c=1 -> mean[1] (G mean), c=2 -> mean[0] (B mean).
                    int bgrCh = c == 0 ? chR : (c == 1 ? chG : chB);
                    int meanIdx = c == 0 ? 2 : (c == 1 ? 1 : 0);
                    result[bgrCh + dstIdx] = val - mean[meanIdx];
                }
            }
        }

        return result;
    }

    private static byte[] ResizeRgb(byte[] srcRgb, int srcW, int srcH, int dstW, int dstH)
    {
        float scaleX = (float)srcW / dstW;
        float scaleY = (float)srcH / dstH;
        var result = new byte[dstW * dstH * 3];

        for (int dy = 0; dy < dstH; dy++)
        {
            float sy = dy * scaleY;
            for (int dx = 0; dx < dstW; dx++)
            {
                float sx = dx * scaleX;
                int dst = (dy * dstW + dx) * 3;

                for (int c = 0; c < 3; c++)
                {
                    float val = SampleBilinearChannel(srcRgb, srcW, srcH, sx, sy, c);
                    result[dst + c] = (byte)Math.Clamp((int)MathF.Round(val), 0, 255);
                }
            }
        }

        return result;
    }

    private static float SampleBilinearChannel(byte[] srcRgb, int srcW, int srcH, float sx, float sy, int channel)
    {
        int x0 = Math.Clamp((int)sx, 0, srcW - 1);
        int x1 = Math.Min(x0 + 1, srcW - 1);
        int y0 = Math.Clamp((int)sy, 0, srcH - 1);
        int y1 = Math.Min(y0 + 1, srcH - 1);
        float fx = sx - x0;
        float fy = sy - y0;

        int srcIdx00 = (y0 * srcW + x0) * 3 + channel;
        int srcIdx10 = (y0 * srcW + x1) * 3 + channel;
        int srcIdx01 = (y1 * srcW + x0) * 3 + channel;
        int srcIdx11 = (y1 * srcW + x1) * 3 + channel;

        float v00 = srcRgb[srcIdx00];
        float v10 = srcRgb[srcIdx10];
        float v01 = srcRgb[srcIdx01];
        float v11 = srcRgb[srcIdx11];
        float top = (1 - fx) * v00 + fx * v10;
        float bottom = (1 - fx) * v01 + fx * v11;
        return (1 - fy) * top + fy * bottom;
    }

    private static void ValidateChannels(float[] mean, float[] stddev)
    {
        if (mean.Length != 3 || stddev.Length != 3)
        {
            throw new ArgumentException("mean and stddev must both be arrays of length 3.");
        }
    }
}
