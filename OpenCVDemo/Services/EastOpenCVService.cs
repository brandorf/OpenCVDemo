using System.Diagnostics;
using OpenCVDemo.ViewModels;
using OpenCvSharp.Dnn;
using OpenCvSharp;
using Point = OpenCvSharp.Point;
using Rect = OpenCvSharp.Rect;
using Size = OpenCvSharp.Size;
using Microsoft.Extensions.Options;

namespace OpenCVDemo.Services;

public class EastOpenCvService : IVideoProcessingService
{
    private readonly DetectionSettings _detectionSettings;
    private int _currentFrame = 0;
    private int _lastFrame = 1;
    private TimeSpan _frameTime = TimeSpan.Zero;
    private readonly EastOpenCvServiceConfiguration _config;

    public EastOpenCvService(IOptions<EastOpenCvServiceConfiguration> config)
    {
       // _detectionSettings = detectionSettings;
        _config = config.Value;
        Detections = new List<Detection>();
    }

    public async Task ProcessVideo(string videoFilePath)
    {
        Cv2.SetNumThreads(8);

        Detections.Clear();

        // Create a VideoCapture object from the video file
        using var video = new VideoCapture(videoFilePath);

        var modelCombinedPath = Path.Combine(Path.Combine(AppContext.BaseDirectory, "Resources"), _config.ModelPath);

        // Load the pre-trained EAST model for text detection
        var net = CvDnn.ReadNet(modelCombinedPath);

        CurrentFrame = 0;
        LastFrame = (int)video.FrameCount;
        // Initialize the stopwatch
        var stopwatch = new Stopwatch();

        Mat previousFrame = null;
        // Process each frame of the video
        using var frame = new Mat();
        while (true)
        {
            // Start the stopwatch
            stopwatch.Start();

            // Read the next frame from the video
            // If the frame is null, then we have reached the end of the video
            if (!video.Read(frame))
            {
                break;
            }

            CurrentFrame = (int)video.PosFrames;

            // If this is not the first frame, and it is similar to the previous one, skip it
            if (previousFrame == null || !AreFramesSimilar(frame, previousFrame))
            {
                var newDetection = ProcessSingleFrame(frame);
                
                if (Detections.Any())
                {
                    var lastDetection = Detections.Last();
                    if (!AreSimilar(newDetection, lastDetection))
                    {
                        Detections.Add(newDetection);
                        DetectionsChanged?.Invoke(newDetection);
                    }
                }
                else
                {
                    Detections.Add(newDetection);
                    DetectionsChanged?.Invoke(newDetection);
                }
                
                previousFrame = frame.Clone();
            }

            stopwatch.Stop();

                _frameTime = stopwatch.Elapsed;
                // Calculate the frames per second
                Fps = 1.0M / (decimal)stopwatch.Elapsed.TotalSeconds;

                // Reset the stopwatch for the next frame
                stopwatch.Reset();
            
        }
    }

    public Detection ProcessSingleFrame(ImageSource image, float? confidenceOverride = null, Scalar? colorOverride = null)
    {
        return ProcessSingleFrame(ImageSourceToMat(image), confidenceOverride, colorOverride);
    }
    
    public Mat ImageSourceToMat(ImageSource imageSource)
    {
        if (imageSource is StreamImageSource streamImageSource)
        {
            // Get the stream from the StreamImageSource
            using Stream imageStream = ((IStreamImageSource)streamImageSource).GetStreamAsync().ConfigureAwait(false).GetAwaiter().GetResult();
           if (imageStream == null) return null;

            // Convert stream to byte array
            using var memoryStream = new MemoryStream();
            imageStream.CopyTo(memoryStream);
            byte[] imageData = memoryStream.ToArray();

            // Decode the byte array to a Mat object
            Mat imageMat = Cv2.ImDecode(imageData, ImreadModes.Color);

            return imageMat;
        }
        else
        {
            // Handle other types of ImageSource if necessary
            throw new NotImplementedException("Only StreamImageSource is supported in this example.");
        }
    }
    
    private Detection ProcessSingleFrame(Mat frame, float? confidenceOverride = null, Scalar? colorOverride = null)
    {
        if (frame.Empty())
        {
            throw new ArgumentException("The frame is empty.", nameof(frame));
        }

        int originalWidth = frame.Width;
        int originalHeight = frame.Height;
        int width = originalWidth - (originalWidth % 32);
        int height = originalHeight - (originalHeight % 32);

        Mat resizedFrame = new Mat();
        Cv2.Resize(frame, resizedFrame, new Size(width, height));

        var blob = CvDnn.BlobFromImage(resizedFrame);
        var net = CvDnn.ReadNet(Path.Combine(AppContext.BaseDirectory, "Resources", _config.ModelPath));
        
        // Set the backend and target for the network
        net.SetPreferableBackend(Backend.VKCOM);
        net.SetPreferableTarget(Target.CPU); // or Target.CUDA for CUDA target

        // Check the backend and target

        //bool isHardwareAccelerated = (backend == Backend.CUDA || backend == Backend.OPENCV) &&
        //                             (target == Target.CUDA || target == Target.OPENCL);

        //Console.WriteLine($"Is hardware acceleration enabled: {isHardwareAccelerated}");        
        
        
        net.SetInput(blob);
        net.SetPreferableBackend(Backend.OPENCV);
        net.SetPreferableTarget(Target.OPENCL);

        var scores = net.Forward("feature_fusion/Conv_7/Sigmoid");
        var geometry = net.Forward("feature_fusion/concat_3");

        var boxes = GetBoundingBoxes(resizedFrame, scores, geometry, confidenceOverride.HasValue ? confidenceOverride.Value : _config.ConfidenceThreshold);

        var textRegions = new List<Rect>();
        foreach (var box in boxes)
        {
            textRegions.Add(box);
            Cv2.Rectangle(resizedFrame, box, colorOverride.HasValue ? colorOverride.Value : Scalar.Green, 2);
        }

        return new Detection { Frame = resizedFrame, BoundingBoxes = new List<Rect>(textRegions) };
    }

    public List<Detection> Detections { get; private set; }

    public decimal FPS { get; }
    public event Action ProgressChanged;
    public event Action<Detection>? DetectionsChanged;

    public decimal ProgressPercent
    {
        get => (decimal)CurrentFrame / LastFrame;
        private set
        {
            if (value != ProgressPercent)
            {
                ProgressChanged?.Invoke();
            }
        }
    }

    public int CurrentFrame
    {
        get => _currentFrame;
        set
        {
            _currentFrame = value;
            ProgressChanged?.Invoke();
        }
    }

    public decimal Fps { get; private set; }

    public int LastFrame
    {
        get => _lastFrame;
        set
        {
            _lastFrame = value;
            ProgressChanged?.Invoke();
        }
    }

    public TimeSpan FrameTime
    {
        get => _frameTime;
        set { _frameTime = value; }
    }

    private List<Rect> GetBoundingBoxes(Mat frame, Mat scores, Mat geometry, float scoreThresh)
    {
        List<Rect> boundingBoxes = new List<Rect>();
        List<float> confidences = new List<float>();

        for (int y = 0; y < scores.Size(2); y++)
        {
            for (int x = 0; x < scores.Size(3); x++)
            {
                float score = scores.At<float>(0, 0, y, x);

                // If our score does not have sufficient probability, ignore it
                if (score < scoreThresh)
                    continue;

                // Compute the offset factor as our resulting feature maps will
                // be 4x smaller than the input image
                float offsetX = x * 4.0f;
                float offsetY = y * 4.0f;

                // Extract the rotation angle for the prediction and then
                // compute the sin and cosine
                float angle = geometry.At<float>(0, 4, y, x);
                float cos = (float)Math.Cos(angle);
                float sin = (float)Math.Sin(angle);

                // Use the geometry volume to derive the width and height of
                // the bounding box
                float h = geometry.At<float>(0, 0, y, x) + geometry.At<float>(0, 2, y, x);
                float w = geometry.At<float>(0, 1, y, x) + geometry.At<float>(0, 3, y, x);

                // Compute both the starting and ending (x, y)-coordinates
                // for the text prediction bounding box
                Point2f offset = new Point2f(
                    offsetX + cos * geometry.At<float>(0, 1, y, x) - sin * geometry.At<float>(0, 2, y, x),
                    offsetY + sin * geometry.At<float>(0, 1, y, x) + cos * geometry.At<float>(0, 2, y, x));
                Point2f p1 = new Point2f(-sin * h, -cos * h) + offset;
                Point2f p3 = new Point2f(-cos * w, sin * w) + offset;
                RotatedRect r = new RotatedRect((p1 + p3) * 0.5f, new Size2f(w, h), -angle * 180.0f / (float)Math.PI);

                // Get the bounding rectangle of the rotated rectangle
                Rect boundingRect = r.BoundingRect();

                // Ensure the bounding rectangle is within the frame
                boundingRect = Rect.Intersect(boundingRect, new Rect(0, 0, frame.Width, frame.Height));

                // If the bounding rectangle is empty after the intersection, skip this box
                if (boundingRect.Width <= 0 || boundingRect.Height <= 0)
                    continue;

                boundingBoxes.Add(boundingRect);
                confidences.Add(score);
            }
        }

        // Apply non-maximum suppression
        CvDnn.NMSBoxes(boundingBoxes, confidences, scoreThresh, _config.NMSThreshold, out var indices);

        // Return only the bounding boxes that were not suppressed
        return indices.Select(index => boundingBoxes[index]).ToList();
    }

    private bool AreSimilar(Detection detection1, Detection detection2)
    {
        // Compare the number of bounding boxes
        if (detection1.BoundingBoxes.Count != detection2.BoundingBoxes.Count)
        {
            return false;
        }

        // Compare each bounding box
        for (int i = 0; i < detection1.BoundingBoxes.Count; i++)
        {
            if (!BoundingBoxesEquivalent(detection1.BoundingBoxes, detection2.BoundingBoxes))
            {
                return false;
            }
        }

        return true;
    }

    private bool BoundingBoxesEquivalent( IList<Rect> first, IList<Rect> second)
    {
        // If the arrays are not the same length, they are not equal
        if (first.Count != second.Count)
        {
            return false;
        }

        // Compare each Rect in the arrays
        for (int i = 0; i < first.Count; i++)
        {
            if (first[i].X != second[i].X || first[i].Y != second[i].Y ||
                first[i].Width != second[i].Width || first[i].Height != second[i].Height)
            {
                return false;
            }
        }

        // If all Rects are equal, the arrays are equal
        return true;
    }

    public bool AreFramesSimilar(Mat frame1, Mat frame2)
    {
        using var diff = new Mat();
        Cv2.Absdiff(frame1, frame2, diff);
        Scalar avg = Cv2.Mean(diff);
        return avg.Val0 < _config.FrameSimilarityThreshold;
    }


}