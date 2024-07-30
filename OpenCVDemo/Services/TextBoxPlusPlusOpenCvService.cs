using System.Diagnostics;
using OpenCVDemo.ViewModels;
using OpenCvSharp.Dnn;
using OpenCvSharp;
using Point = OpenCvSharp.Point;
using Rect = OpenCvSharp.Rect;
using Size = OpenCvSharp.Size;
using Microsoft.Extensions.Options;

namespace OpenCVDemo.Services;

public class TextBoxPlusPlusOpenCvService : IVideoProcessingService
{
    private int _currentFrame = 0;
    private int _lastFrame = 1;
    private TimeSpan _frameTime = TimeSpan.Zero;
    private readonly TextBoxPlusPlusOpenCvServiceConfiguration _config;


    public TextBoxPlusPlusOpenCvService(IOptions<TextBoxPlusPlusOpenCvServiceConfiguration> config)
    {
        _config = config.Value;
        Detections = new List<Detection>();
    }

    public Detection ProcessSingleFrame(ImageSource image, float? confidenceOverride = null, Scalar? colorOverride = null)
    {
        throw new NotImplementedException();
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

    public async Task ProcessVideo(string videoFilePath)
    {
        Cv2.SetNumThreads(8);

        Detections.Clear();

        // Create a VideoCapture object from the video file
        using var video = new VideoCapture(videoFilePath);

        var modelCombinedPath = Path.Combine(Path.Combine(AppContext.BaseDirectory, "Resources"), _config.ModelPath);
        var prototextPath = Path.Combine(Path.Combine(AppContext.BaseDirectory, "Resources"), _config.PrototextPath);

        VerifyModelFileAccess(modelCombinedPath);

        // Load the pre-trained TextBox++ model for text detection
        var net = CvDnn.ReadNetFromCaffe(prototextPath, modelCombinedPath);

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
            if (!video.Read(frame) || frame.Empty())
            {
                break;
            }

            CurrentFrame = (int)video.PosFrames;

            // If this is not the first frame and it is similar to the previous one, skip it
            if (previousFrame == null || !AreFramesSimilar(frame, previousFrame))
            {
                // Convert the frame to a blob to be used as input for the TextBoxes++ model
                var blob = CvDnn.BlobFromImage(frame, 1, new Size(300, 300), new Scalar(104, 117, 123), swapRB: false, crop: false);

                if(blob.Empty())
                {
                    throw new InvalidOperationException("Hmm.");
                }

                // Pass the blob through the network and obtain the detections and predictions
                net.SetInput(blob);
                net.SetPreferableBackend(Backend.OPENCV);
                net.SetPreferableTarget(Target.OPENCL);
                var detections = net.Forward();

                var boxes = GetBoundingBoxes(detections);

                var textRegions = new List<Rect>();
                foreach (var box in boxes)
                {
                    textRegions.Add(box);
                    Cv2.Rectangle(frame, box, Scalar.Green, 2);
                }

                var newDetection = new Detection { Frame = frame, BoundingBoxes = new List<Rect>(textRegions) };

                // If Detections list is not empty, compare newDetection with the last one
                if (Detections.Any())
                {
                    var lastDetection = Detections.Last();

                    // Define your comparison logic here. This is just a simple example.
                    if (AreSimilar(newDetection, lastDetection))
                    {
                        continue;
                    }
                }

                Detections.Add(newDetection);
                DetectionsChanged?.Invoke(newDetection);

                previousFrame = frame.Clone();
                // Release the blob to free up memory
                blob.Dispose();

            }

            stopwatch.Stop();

            _frameTime = stopwatch.Elapsed;
            // Calculate the frames per second
            Fps = 1.0M / (decimal)stopwatch.Elapsed.TotalSeconds;

            // Reset the stopwatch for the next frame
            stopwatch.Reset();
        }
    }

    private void VerifyModelFileAccess(string modelCombinedPath)
    {
        // Check if the model file exists
        if (!File.Exists(modelCombinedPath))
        {
            throw new FileNotFoundException($"The model file at {modelCombinedPath} does not exist.");
        }

        // Check if we have read permissions for the model file
        try
        {
            using var stream = File.OpenRead(modelCombinedPath);
        }
        catch (UnauthorizedAccessException)
        {
            throw new UnauthorizedAccessException($"Read permission for the model file at {modelCombinedPath} is denied.");
        }
    }

    // Other methods and properties remain the same as in the original OpenCvService class
    // ...

    private List<Rect> GetBoundingBoxes(Mat detections)
    {
        List<Rect> boxes = new List<Rect>();

        int numDetections = detections.Size(2);
        for (int i = 0; i < numDetections; i++)
        {
            float centerX = detections.At<float>(0, 0, i, 0);
            float centerY = detections.At<float>(0, 0, i, 1);
            float width = detections.At<float>(0, 0, i, 2);
            float height = detections.At<float>(0, 0, i, 3);
            float angle = detections.At<float>(0, 0, i, 4);

            // Convert angle from radians to degrees
            angle = angle * 180.0f / (float)Math.PI;

            // Create a rotated rectangle from the TextBox++ output parameters
            RotatedRect rotatedRect = new RotatedRect(new Point2f(centerX, centerY), new Size2f(width, height), angle);

            // Get the bounding rectangle of the rotated rectangle
            Rect boundingRect = rotatedRect.BoundingRect();

            boxes.Add(boundingRect);
        }

        return boxes;
    }

    public bool AreFramesSimilar(Mat frame1, Mat frame2)
    {
        using var diff = new Mat();
        Cv2.Absdiff(frame1, frame2, diff);
        Scalar avg = Cv2.Mean(diff);
        return avg.Val0 < 5;
    }

    private bool AreSimilar(Detection detection1, Detection detection2)
    {
        // Compare the number of bounding boxes
        if (detection1.BoundingBoxes.Count != detection2.BoundingBoxes.Count)
        {
            return false;
        }

        //// Compare each bounding box
        //for (int i = 0; i < detection1.BoundingBoxes.Count; i++)
        //{
        //    if (!BoundingBoxesEquivalent(detection1.BoundingBoxes, detection2.BoundingBoxes))
        //    {
        //        return false;
        //    }
        //}

        return true;
    }
}
