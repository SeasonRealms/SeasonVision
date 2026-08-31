// Copyright (c) SeasonEngine and contributors.
// Licensed under the MIT License.
// https://github.com/SeasonRealms/SeasonVision

//https://github.com/yakhyo/pipnet-onnx

namespace Season.Vision;

/// <summary>
/// PIPNet ONNX face landmark detection.
/// Uses Ultraface face boxes, then applies the PIPNet crop, ImageNet
/// normalization, and five-tensor decoding pipeline to recover landmarks.
/// </summary>
public static class FaceLandmark
{
    private const int NumNeighbors = 10;
    private const float CropPadding = 0.1f;
    private static readonly float[] ImageNetMean = { 0.485f, 0.456f, 0.406f };
    private static readonly float[] ImageNetStd = { 0.229f, 0.224f, 0.225f };

    public static FaceLandmarkResult Detect(
        InferenceSession detectorSession,
        InferenceSession recognizerSession,
        ReadOnlySpan<byte> imageData, int width, int height,
        bool createAnnotatedImage = false,
        bool drawConnections = false,
        int maxFaces = 5)
    {

        if (maxFaces <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxFaces), "maxFaces must be greater than 0.");

        var config = GetModelConfig(recognizerSession);
        var reverseInfo = PIPNetMeanface.GetReverseInfo(config.NumLandmarks, NumNeighbors);
        var detectorFaces = Ultraface.Detect(detectorSession, imageData, width, height, false, maxFaces);
        var rgb = ImageProcessor.ExtractRgb(imageData, width, height);

        var result = new FaceLandmarkResult
        {
            Model = null, //model,
            DetectorModel = null, //detectorModel,
            ImageWidth = width,
            ImageHeight = height,
            RequestedMaxFaces = maxFaces,
            LandmarkCount = config.NumLandmarks,
            InputWidth = config.InputWidth,
            InputHeight = config.InputHeight
        };

        for (int i = 0; i < detectorFaces.Faces.Count; i++)
        {
            var detectedFace = detectorFaces.Faces[i];
            var crop = ExpandCrop(detectedFace.Box, width, height);
            var cropRgb = ImageProcessor.CropRgb(rgb, width, height, crop.X, crop.Y, crop.Width, crop.Height);
            float[] chw = ImageProcessor.ResizeNormalizeToNchw(
                cropRgb,
                crop.Width,
                crop.Height,
                config.InputWidth,
                config.InputHeight,
                ImageNetMean,
                ImageNetStd,
                scaleToUnitInterval: true);

            var input = new DenseTensor<float>(chw, new[] { 1, 3, config.InputHeight, config.InputWidth });
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(config.InputName, input)
            };

            using var outputs = recognizerSession.Run(inputs);
            var landmarks = Decode(outputs, config, reverseInfo, crop);

            result.Faces.Add(new FaceLandmarkFace
            {
                Index = i,
                Confidence = detectedFace.Confidence,
                BoundingBox = new FaceLandmarkBox(
                    detectedFace.Box.Xmin,
                    detectedFace.Box.Ymin,
                    detectedFace.Box.Xmax,
                    detectedFace.Box.Ymax),
                CropBox = new FaceLandmarkBox(
                    crop.X,
                    crop.Y,
                    crop.X + crop.Width - 1,
                    crop.Y + crop.Height - 1),
                Landmarks = landmarks
            });
        }

        if (createAnnotatedImage)
            result.AnnotatedImage = DrawResults(imageData, width, height, result.Faces, drawConnections);

        return result;
    }

    private static byte[] DrawResults(ReadOnlySpan<byte> imageData, int width, int height, List<FaceLandmarkFace> faces, bool drawConnections)
    {
        if (faces.Count == 0)
            return imageData.ToArray();

        var pixels = ImageProcessor.EnsureRgba(imageData, width, height);
        int pointRadius = Math.Max(1, Math.Min(width, height) / 320);
        int boxThickness = Math.Max(2, Math.Min(width, height) / 300);
        int lineThickness = Math.Max(1, Math.Min(width, height) / 500);

        foreach (var face in faces)
        {
            DrawRectOutline(pixels, width, height, face.BoundingBox, boxThickness, 255, 0, 0);

            if (drawConnections)
                DrawConnections(pixels, width, height, face.Landmarks, lineThickness, 0, 180, 255);

            foreach (var point in face.Landmarks)
                FillCircle(pixels, width, height, point.X, point.Y, pointRadius, 0, 255, 0);
        }

        return pixels;
    }

    private static List<FaceLandmarkPoint> Decode(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs,
        PipNetModelConfig config,
        PipNetReverseInfo reverseInfo,
        FaceCrop crop)
    {
        var tensors = outputs.ToArray();
        if (tensors.Length < 5)
            throw new InvalidOperationException($"Unexpected number of PIPNet output tensors. Expected 5, actual: {tensors.Length}.");

        float[] clsMap = tensors[0].AsEnumerable<float>().ToArray();
        float[] offsetX = tensors[1].AsEnumerable<float>().ToArray();
        float[] offsetY = tensors[2].AsEnumerable<float>().ToArray();
        float[] nbX = tensors[3].AsEnumerable<float>().ToArray();
        float[] nbY = tensors[4].AsEnumerable<float>().ToArray();

        int numLandmarks = config.NumLandmarks;
        int featH = config.FeatureHeight;
        int featW = config.FeatureWidth;
        int plane = featH * featW;
        float scaleX = (float)config.InputWidth / config.NetStride;
        float scaleY = (float)config.InputHeight / config.NetStride;

        var predX = new float[numLandmarks];
        var predY = new float[numLandmarks];
        var nbPredX = new float[numLandmarks, NumNeighbors];
        var nbPredY = new float[numLandmarks, NumNeighbors];

        for (int landmarkIndex = 0; landmarkIndex < numLandmarks; landmarkIndex++)
        {
            int clsBase = landmarkIndex * plane;
            int maxId = 0;
            float maxScore = float.MinValue;
            for (int idx = 0; idx < plane; idx++)
            {
                float score = clsMap[clsBase + idx];
                if (score > maxScore)
                {
                    maxScore = score;
                    maxId = idx;
                }
            }

            int col = maxId % featW;
            int row = maxId / featW;
            predX[landmarkIndex] = (col + offsetX[clsBase + maxId]) / scaleX;
            predY[landmarkIndex] = (row + offsetY[clsBase + maxId]) / scaleY;

            for (int neighborIndex = 0; neighborIndex < NumNeighbors; neighborIndex++)
            {
                int neighborBase = ((landmarkIndex * NumNeighbors) + neighborIndex) * plane;
                nbPredX[landmarkIndex, neighborIndex] = (col + nbX[neighborBase + maxId]) / scaleX;
                nbPredY[landmarkIndex, neighborIndex] = (row + nbY[neighborBase + maxId]) / scaleY;
            }
        }

        var points = new List<FaceLandmarkPoint>(numLandmarks);
        for (int landmarkIndex = 0; landmarkIndex < numLandmarks; landmarkIndex++)
        {
            float sumX = predX[landmarkIndex];
            float sumY = predY[landmarkIndex];
            int reverseBase = landmarkIndex * reverseInfo.MaxLen;

            for (int k = 0; k < reverseInfo.MaxLen; k++)
            {
                int fromLandmark = reverseInfo.ReverseIndex1[reverseBase + k];
                int fromNeighbor = reverseInfo.ReverseIndex2[reverseBase + k];
                sumX += nbPredX[fromLandmark, fromNeighbor];
                sumY += nbPredY[fromLandmark, fromNeighbor];
            }

            float mergedX = Math.Clamp(sumX / (reverseInfo.MaxLen + 1), 0f, 1f);
            float mergedY = Math.Clamp(sumY / (reverseInfo.MaxLen + 1), 0f, 1f);

            points.Add(new FaceLandmarkPoint
            {
                Index = landmarkIndex,
                X = mergedX * crop.Width + crop.X,
                Y = mergedY * crop.Height + crop.Y
            });
        }

        return points;
    }

    private static FaceCrop ExpandCrop(UltrafaceBox box, int imageWidth, int imageHeight)
    {
        float faceWidth = box.Xmax - box.Xmin + 1f;
        float faceHeight = box.Ymax - box.Ymin + 1f;

        int x1 = Math.Max(0, (int)(box.Xmin - faceWidth * CropPadding));
        int y1 = Math.Max(0, (int)(box.Ymin + faceHeight * CropPadding));
        int x2 = Math.Min(imageWidth - 1, (int)(box.Xmax + faceWidth * CropPadding));
        int y2 = Math.Min(imageHeight - 1, (int)(box.Ymax + faceHeight * CropPadding));

        return new FaceCrop(
            x1,
            y1,
            Math.Max(1, x2 - x1 + 1),
            Math.Max(1, y2 - y1 + 1));
    }

    private static PipNetModelConfig GetModelConfig(InferenceSession session)
    {
        var inputMetadata = session.InputMetadata.First();
        int inputHeight = ResolveDimension(inputMetadata.Value.Dimensions, 2, 256);
        int inputWidth = ResolveDimension(inputMetadata.Value.Dimensions, 3, 256);

        var clsOutput = session.OutputMetadata
            .FirstOrDefault(item =>
            {
                int channels = ResolveDimension(item.Value.Dimensions, 1, 0);
                return channels == 68 || channels == 98;
            });

        if (string.IsNullOrWhiteSpace(clsOutput.Key))
            throw new NotSupportedException("Unable to identify the PIPNet cls_map output from the ONNX output metadata.");

        int numLandmarks = ResolveDimension(clsOutput.Value.Dimensions, 1, 0);
        int featureHeight = ResolveDimension(clsOutput.Value.Dimensions, 2, inputHeight / 32);
        int featureWidth = ResolveDimension(clsOutput.Value.Dimensions, 3, inputWidth / 32);

        if (numLandmarks != 68 && numLandmarks != 98)
            throw new NotSupportedException($"Only 68-point and 98-point PIPNet models are supported. Actual output channel count: {numLandmarks}.");

        return new PipNetModelConfig(
            inputMetadata.Key,
            inputWidth,
            inputHeight,
            numLandmarks,
            featureWidth,
            featureHeight);
    }

    private static int ResolveDimension(IReadOnlyList<int> dimensions, int index, int fallback)
    {
        if (index < dimensions.Count && dimensions[index] > 0)
            return dimensions[index];

        return fallback;
    }

    private static void DrawConnections(
        byte[] pixels,
        int imageWidth,
        int imageHeight,
        List<FaceLandmarkPoint> landmarks,
        int thickness,
        byte r,
        byte g,
        byte b)
    {
        if (landmarks.Count == 68)
        {
            DrawPolyline(pixels, imageWidth, imageHeight, landmarks, 0, 16, false, thickness, r, g, b);
            DrawPolyline(pixels, imageWidth, imageHeight, landmarks, 17, 21, false, thickness, r, g, b);
            DrawPolyline(pixels, imageWidth, imageHeight, landmarks, 22, 26, false, thickness, r, g, b);
            DrawPolyline(pixels, imageWidth, imageHeight, landmarks, 27, 30, false, thickness, r, g, b);
            DrawPolyline(pixels, imageWidth, imageHeight, landmarks, 31, 35, false, thickness, r, g, b);
            DrawPolyline(pixels, imageWidth, imageHeight, landmarks, 36, 41, true, thickness, r, g, b);
            DrawPolyline(pixels, imageWidth, imageHeight, landmarks, 42, 47, true, thickness, r, g, b);
            DrawPolyline(pixels, imageWidth, imageHeight, landmarks, 48, 59, true, thickness, r, g, b);
            DrawPolyline(pixels, imageWidth, imageHeight, landmarks, 60, 67, true, thickness, r, g, b);
            return;
        }

        if (landmarks.Count == 98)
        {
            DrawPolyline(pixels, imageWidth, imageHeight, landmarks, 0, 32, false, thickness, r, g, b);
            DrawPolyline(pixels, imageWidth, imageHeight, landmarks, 33, 41, false, thickness, r, g, b);
            DrawPolyline(pixels, imageWidth, imageHeight, landmarks, 42, 50, false, thickness, r, g, b);
            DrawPolyline(pixels, imageWidth, imageHeight, landmarks, 51, 59, false, thickness, r, g, b);
            DrawPolyline(pixels, imageWidth, imageHeight, landmarks, 60, 67, true, thickness, r, g, b);
            DrawPolyline(pixels, imageWidth, imageHeight, landmarks, 68, 75, true, thickness, r, g, b);
            DrawPolyline(pixels, imageWidth, imageHeight, landmarks, 76, 87, true, thickness, r, g, b);
            DrawPolyline(pixels, imageWidth, imageHeight, landmarks, 88, 95, true, thickness, r, g, b);
        }
    }

    private static void DrawPolyline(
        byte[] pixels,
        int imageWidth,
        int imageHeight,
        List<FaceLandmarkPoint> landmarks,
        int startIndex,
        int endIndex,
        bool closed,
        int thickness,
        byte r,
        byte g,
        byte b)
    {
        if (startIndex < 0 || endIndex >= landmarks.Count || endIndex <= startIndex)
            return;

        for (int i = startIndex; i < endIndex; i++)
        {
            DrawLine(
                pixels,
                imageWidth,
                imageHeight,
                landmarks[i].X,
                landmarks[i].Y,
                landmarks[i + 1].X,
                landmarks[i + 1].Y,
                thickness,
                r,
                g,
                b);
        }

        if (closed)
        {
            DrawLine(
                pixels,
                imageWidth,
                imageHeight,
                landmarks[endIndex].X,
                landmarks[endIndex].Y,
                landmarks[startIndex].X,
                landmarks[startIndex].Y,
                thickness,
                r,
                g,
                b);
        }
    }

    private static void DrawLine(
        byte[] pixels,
        int imageWidth,
        int imageHeight,
        float x0,
        float y0,
        float x1,
        float y1,
        int thickness,
        byte r,
        byte g,
        byte b)
    {
        int steps = Math.Max(Math.Abs((int)MathF.Round(x1 - x0)), Math.Abs((int)MathF.Round(y1 - y0)));
        steps = Math.Max(steps, 1);

        for (int i = 0; i <= steps; i++)
        {
            float t = i / (float)steps;
            float x = x0 + (x1 - x0) * t;
            float y = y0 + (y1 - y0) * t;
            FillCircle(pixels, imageWidth, imageHeight, x, y, thickness, r, g, b);
        }
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

    private static void FillCircle(
        byte[] pixels,
        int imageWidth,
        int imageHeight,
        float centerX,
        float centerY,
        int radius,
        byte r,
        byte g,
        byte b)
    {
        int cx = (int)MathF.Round(centerX);
        int cy = (int)MathF.Round(centerY);
        int radiusSquared = radius * radius;

        for (int y = cy - radius; y <= cy + radius; y++)
        {
            for (int x = cx - radius; x <= cx + radius; x++)
            {
                int dx = x - cx;
                int dy = y - cy;
                if (dx * dx + dy * dy <= radiusSquared)
                    SetPixel(pixels, imageWidth, imageHeight, x, y, r, g, b, 255);
            }
        }
    }

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

    private sealed record PipNetModelConfig(
        string InputName,
        int InputWidth,
        int InputHeight,
        int NumLandmarks,
        int FeatureWidth,
        int FeatureHeight)
    {
        public int NetStride => InputHeight / FeatureHeight;
    }

    private sealed record PipNetReverseInfo(int[] ReverseIndex1, int[] ReverseIndex2, int MaxLen);

    private sealed record FaceCrop(int X, int Y, int Width, int Height);

    private static class PIPNetMeanface
    {
        private static readonly float[] Meanface300W68 =
        {
            0.05558998895410058f, 0.23848280098218655f, 0.05894856684324656f, 0.3590187767402909f,
            0.0736574254414371f, 0.4792196439871159f, 0.09980016420365162f, 0.5959029676167197f,
            0.14678670154995865f, 0.7035615597409001f, 0.21847188218752928f, 0.7971705893013413f,
            0.30554692814599393f, 0.8750572978073209f, 0.4018434142644611f, 0.9365018059444535f,
            0.5100536090382116f, 0.9521295666029498f, 0.6162039414413925f, 0.9309467340899419f,
            0.7094522484942942f, 0.8669275031738761f, 0.7940993502957612f, 0.7879369615524398f,
            0.8627063649669019f, 0.6933756633633967f, 0.9072386130534111f, 0.5836975017700834f,
            0.9298874997796132f, 0.4657004930314701f, 0.9405202670724796f, 0.346063993805527f,
            0.9425419553088846f, 0.22558131891345742f, 0.13304298285530403f, 0.14853071838028062f,
            0.18873587368440375f, 0.09596491613770254f, 0.2673231915839219f, 0.08084218279128136f,
            0.34878638553224905f, 0.09253591849498964f, 0.4226713753717798f, 0.12466063383809506f,
            0.5618513152452376f, 0.11839668911898667f, 0.6394952560845826f, 0.08480191391770678f,
            0.7204375851516752f, 0.07249669092117161f, 0.7988615904537885f, 0.08766933146893043f,
            0.8534884939460948f, 0.1380096813348583f, 0.49610677423740546f, 0.21516740699375395f,
            0.49709661403980665f, 0.2928875699060973f, 0.4982292618461611f, 0.3699985379939941f,
            0.49982965173254235f, 0.4494119144493957f, 0.406772397599095f, 0.5032397294041786f,
            0.45231994786363067f, 0.5197953144002292f, 0.49969685987914064f, 0.5332489262413073f,
            0.5470074224053442f, 0.518413595827126f, 0.5892261151542287f, 0.5023530079850803f,
            0.22414578747180394f, 0.22835847349949062f, 0.27262947128194215f, 0.19915251892241678f,
            0.3306759252861797f, 0.20026034220607236f, 0.38044435864341913f, 0.23839196034290633f,
            0.32884072789429913f, 0.24902443794896897f, 0.2707409300714473f, 0.24950886025380967f,
            0.6086826011068529f, 0.23465048639345917f, 0.660397116846103f, 0.1937087938594717f,
            0.7177815187666494f, 0.19317079039835858f, 0.7652328176062365f, 0.22088822845258235f,
            0.722727677909097f, 0.24195514178450958f, 0.6658378927310327f, 0.2441554205021945f,
            0.32894370935769124f, 0.6496589505331646f, 0.39347179739100613f, 0.6216899667490776f,
            0.4571976492475472f, 0.60794251109236f, 0.4990484623797022f, 0.6190124015360254f,
            0.5465555522325872f, 0.6071477960565326f, 0.6116127327356168f, 0.6205387097430033f,
            0.6742318496058836f, 0.6437466364395467f, 0.6144773141699744f, 0.7077526646009754f,
            0.5526442055374252f, 0.7363350735898412f, 0.5018120662554302f, 0.7424476622366345f,
            0.4554458875556401f, 0.7382303858617719f, 0.3923750731597415f, 0.7118887028663435f,
            0.35530766372404593f, 0.6524479416354049f, 0.457111071610868f, 0.6467108367268608f,
            0.49974082228815025f, 0.6508406774477011f, 0.5477027224368399f, 0.6451242819422733f,
            0.6478392760505715f, 0.647852382880368f, 0.5488474760115958f, 0.6779061893042735f,
            0.5001073351044452f, 0.6845280260362221f, 0.4564831746654594f, 0.6799300301441035f
        };

        private static readonly float[] MeanfaceWflw98 =
        {
            0.07960419395480703f, 0.3921576875344978f, 0.08315055593117261f, 0.43509551571809146f,
            0.08675705281580391f, 0.47810288286566444f, 0.09141892980469117f, 0.5210356946467262f,
            0.09839925903528965f, 0.5637522280060038f, 0.10871037524559955f, 0.6060410614977951f,
            0.12314562992759207f, 0.6475338700558225f, 0.14242389255404694f, 0.6877152027028081f,
            0.16706295456951875f, 0.7259564546408682f, 0.19693946055282413f, 0.761730578566735f,
            0.23131827931527224f, 0.7948205670466106f, 0.2691730934906831f, 0.825332081636482f,
            0.3099415030959131f, 0.853325959406618f, 0.3535202097901413f, 0.8782538906229107f,
            0.40089023799272033f, 0.8984102434399625f, 0.4529251732310723f, 0.9112191359814178f,
            0.5078640056794708f, 0.9146712690731943f, 0.5616519666079889f, 0.9094327772020283f,
            0.6119216923689698f, 0.8950540037623425f, 0.6574617882337107f, 0.8738084866764846f,
            0.6994820494908942f, 0.8482660530943744f, 0.7388135339780575f, 0.8198750461527688f,
            0.775158750479601f, 0.788989141243473f, 0.8078785221990765f, 0.7555462713420953f,
            0.8361052138935441f, 0.7195542055115057f, 0.8592123871172533f, 0.6812759034843933f,
            0.8771159986952748f, 0.6412243940605555f, 0.8902481006481506f, 0.5999743595282084f,
            0.8992952868651163f, 0.5580032282594118f, 0.9050110573289222f, 0.5156548913779377f,
            0.908338439928252f, 0.4731336721500472f, 0.9104896075281127f, 0.4305382486815422f,
            0.9124796341441906f, 0.38798192678294363f, 0.18465941635742913f, 0.35063191749632183f,
            0.24110421889338157f, 0.31190394310826886f, 0.3003235400132397f, 0.30828189837331976f,
            0.3603094923651325f, 0.3135606490643205f, 0.4171060234289877f, 0.32433417646045615f,
            0.416842139562573f, 0.3526729965541497f, 0.36011177591813404f, 0.3439660526998693f,
            0.3000863121140166f, 0.33890077494044946f, 0.24116055928407834f, 0.34065620413845005f,
            0.5709736930161899f, 0.321407825750195f, 0.6305694459247149f, 0.30972642336729495f,
            0.6895161625920927f, 0.3036453838462943f, 0.7488591859761683f, 0.3069143844433495f,
            0.8030471337135181f, 0.3435156012309415f, 0.7485083446528741f, 0.3348759588212388f,
            0.6893025057931884f, 0.33403402013776456f, 0.6304822892126991f, 0.34038458762875695f,
            0.5710009285609654f, 0.34988479902594455f, 0.4954171902473609f, 0.40202330022004634f,
            0.49604903449415433f, 0.4592869389138444f, 0.49644391662771625f, 0.5162862508677217f,
            0.4981161256057368f, 0.5703284628419502f, 0.40749001573145566f, 0.5983629921847019f,
            0.4537396729649631f, 0.6057169923583451f, 0.5007345777827058f, 0.6116695615531077f,
            0.5448481727980428f, 0.6044131443745976f, 0.5882140504891681f, 0.5961738788380111f,
            0.24303324896316683f, 0.40721003719912746f, 0.27771706732644313f, 0.3907171413930685f,
            0.31847706697401107f, 0.38417234007271117f, 0.3621792860449715f, 0.3900847721320633f,
            0.3965299162804086f, 0.41071434661355205f, 0.3586805562211872f, 0.4203724421417311f,
            0.31847860588240934f, 0.4237674602252073f, 0.2789458001651631f, 0.41942757306509065f,
            0.5938514626567266f, 0.4090628827047304f, 0.6303565516542536f, 0.3864501652756091f,
            0.6774844732813035f, 0.3809319896905685f, 0.7150854850525555f, 0.3875173254527522f,
            0.747519807465081f, 0.4025187328459307f, 0.7155172856447009f, 0.4145958479293519f,
            0.680051949453018f, 0.420041513473271f, 0.6359056750107122f, 0.41803782782566573f,
            0.33916483987223056f, 0.6968581311227738f, 0.40008790639758807f, 0.6758101185779204f,
            0.47181947887764153f, 0.6678850445191217f, 0.5025394453374782f, 0.6682917934792593f,
            0.5337748367911458f, 0.6671949030019636f, 0.6015915330083903f, 0.6742535357237751f,
            0.6587068892667173f, 0.6932163943648724f, 0.6192795131720007f, 0.7283129162844936f,
            0.5665923267827963f, 0.7550248076404299f, 0.5031303335863617f, 0.7648348885181623f,
            0.4371030429958871f, 0.7572539606688756f, 0.3814909500115824f, 0.7320595346122074f,
            0.35129809553480984f, 0.6986839074746692f, 0.4247987356100664f, 0.69127609583798f,
            0.5027677238758598f, 0.6911145821740593f, 0.576997542122097f, 0.6896269708051024f,
            0.6471352843446794f, 0.6948977432227927f, 0.5799932528781817f, 0.7185288017567538f,
            0.5024914756021335f, 0.7285408331555782f, 0.4218115644247556f, 0.7209126133193829f,
            0.3219750495122499f, 0.40376441481225156f, 0.6751136343101699f, 0.40023415216110797f
        };

        public static PipNetReverseInfo GetReverseInfo(int numLandmarks, int numNeighbors)
        {
            float[] meanface = numLandmarks switch
            {
                68 => Meanface300W68,
                98 => MeanfaceWflw98,
                _ => throw new NotSupportedException($"Unsupported landmark count: {numLandmarks}")
            };

            int pointCount = meanface.Length / 2;
            var points = new (float X, float Y)[pointCount];
            for (int i = 0; i < pointCount; i++)
                points[i] = (meanface[i * 2], meanface[i * 2 + 1]);

            var neighborIndices = new int[pointCount][];
            for (int i = 0; i < pointCount; i++)
            {
                var distances = new List<(float Distance, int Index)>(pointCount - 1);
                for (int j = 0; j < pointCount; j++)
                {
                    if (i == j)
                        continue;

                    float dx = points[i].X - points[j].X;
                    float dy = points[i].Y - points[j].Y;
                    distances.Add((dx * dx + dy * dy, j));
                }

                neighborIndices[i] = distances
                    .OrderBy(item => item.Distance)
                    .Take(numNeighbors)
                    .Select(item => item.Index)
                    .ToArray();
            }

            var reversed = new (List<int> Index1, List<int> Index2)[pointCount];
            for (int i = 0; i < pointCount; i++)
                reversed[i] = (new List<int>(), new List<int>());

            for (int i = 0; i < pointCount; i++)
            {
                for (int j = 0; j < numNeighbors; j++)
                {
                    int neighbor = neighborIndices[i][j];
                    reversed[neighbor].Index1.Add(i);
                    reversed[neighbor].Index2.Add(j);
                }
            }

            int maxLen = reversed.Max(item => item.Index1.Count);
            var reverseIndex1 = new List<int>(pointCount * maxLen);
            var reverseIndex2 = new List<int>(pointCount * maxLen);

            for (int i = 0; i < pointCount; i++)
            {
                var idx1 = reversed[i].Index1;
                var idx2 = reversed[i].Index2;
                if (idx1.Count == 0)
                {
                    idx1.Add(i);
                    idx2.Add(0);
                }

                while (idx1.Count < maxLen)
                {
                    idx1.Add(idx1[idx1.Count % Math.Max(1, reversed[i].Index1.Count)]);
                    idx2.Add(idx2[idx2.Count % Math.Max(1, reversed[i].Index2.Count)]);
                }

                reverseIndex1.AddRange(idx1.Take(maxLen));
                reverseIndex2.AddRange(idx2.Take(maxLen));
            }

            return new PipNetReverseInfo(reverseIndex1.ToArray(), reverseIndex2.ToArray(), maxLen);
        }
    }
}

public sealed class FaceLandmarkResult
{
    public string Model { get; set; } = string.Empty;
    public string DetectorModel { get; set; } = string.Empty;
    public int ImageWidth { get; set; }
    public int ImageHeight { get; set; }
    public int RequestedMaxFaces { get; set; }
    public int LandmarkCount { get; set; }
    public int InputWidth { get; set; }
    public int InputHeight { get; set; }
    public List<FaceLandmarkFace> Faces { get; set; } = new();
    public byte[] AnnotatedImage { get; set; } = [];

    public string Summary
    {
        get
        {
            return String.Join("\r\n", Faces.Select(fa => $"Index:{fa.Index} Confidence:{fa.Confidence} BoundingBox:{fa.BoundingBox.Xmin} {fa.BoundingBox.Ymin} {fa.BoundingBox.Xmax} {fa.BoundingBox.Ymax} CropBox:{fa.CropBox.Xmin} {fa.CropBox.Ymin} {fa.CropBox.Xmax} {fa.CropBox.Ymax} LandMarks:{fa.Landmarks.Count}"));
        }
    }
}

public sealed class FaceLandmarkFace
{
    public int Index { get; set; }
    public float Confidence { get; set; }
    public FaceLandmarkBox BoundingBox { get; set; } = null!;
    public FaceLandmarkBox CropBox { get; set; } = null!;
    public List<FaceLandmarkPoint> Landmarks { get; set; } = new();
}

public sealed class FaceLandmarkPoint
{
    public int Index { get; set; }
    public float X { get; set; }
    public float Y { get; set; }
}

public sealed class FaceLandmarkBox
{
    public float Xmin { get; set; }
    public float Ymin { get; set; }
    public float Xmax { get; set; }
    public float Ymax { get; set; }

    public FaceLandmarkBox(float xmin, float ymin, float xmax, float ymax)
    {
        Xmin = xmin;
        Ymin = ymin;
        Xmax = xmax;
        Ymax = ymax;
    }
}
