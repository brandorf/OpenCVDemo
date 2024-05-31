using System.Diagnostics;
using OpenCVDemo.ViewModels;
using OpenCvSharp.Dnn;
using OpenCvSharp;
using Point = OpenCvSharp.Point;
using Rect = OpenCvSharp.Rect;
using Size = OpenCvSharp.Size;
using Microsoft.Extensions.Options;

namespace OpenCVDemo.Services;


public class OpenCvService : IVideoProcessingService
{
    private int _currentFrame = 0;
    private int _lastFrame = 1;
    private TimeSpan _frameTime = TimeSpan.Zero;
    private readonly OpenCvServiceConfiguration _config;

    public OpenCvService(IOptions<OpenCvServiceConfiguration> config)
    {
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

        // Process each frame of the video
        using var frame = new Mat();
        while (true)
        {
            // Start the stopwatch
            stopwatch.Start();

            var textRegions = new List<Rect>();


            // Read the next frame from the video
            // If the frame is null, then we have reached the end of the video
            if (!video.Read(frame))
            {
                break;
            }

            CurrentFrame = (int)video.PosFrames;

            Debug.WriteLine($@"Processing Frame {CurrentFrame} of {LastFrame} : {ProgressPercent}%");

            // Convert the frame to a blob to be used as input for the EAST model
            var blob = CvDnn.BlobFromImage(frame, 1.0, new Size(320, 320), new Scalar(123.68, 116.78, 103.94), true,
                false);


            // Pass the blob through the network and obtain the detections and predictions
            net.SetInput(blob);
            net.SetPreferableBackend(Backend.OPENCV);
            net.SetPreferableTarget(Target.OPENCL);
            var scores = net.Forward("feature_fusion/Conv_7/Sigmoid");
            var geometry = net.Forward("feature_fusion/concat_3");


            var boxes = GetBoundingBoxes(frame, scores, geometry, 0.5f);

            foreach (var box in boxes)
            {
                textRegions.Add(box);
                Detections.Add(new Detection { Frame = frame, BoundingBoxes = new List<Rect>(textRegions) });
                DetectionsCHanged?.Invoke();
            }

            // Release the blob to free up memory
            blob.Dispose();

            stopwatch.Stop();

            _frameTime = stopwatch.Elapsed;
            // Calculate the frames per second
            Fps = 1.0M / (decimal)stopwatch.Elapsed.TotalSeconds;

            // Reset the stopwatch for the next frame
            stopwatch.Reset();
        }
    }

    public List<Detection> Detections { get; private set; }

    public decimal FPS { get; }
    public event Action ProgressChanged;
    public event Action? DetectionsChanged;
    public event Action? DetectionsCHanged;

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

                // Draw the bounding rectangle on the frame
                Point[] vertices = Array.ConvertAll(r.Points(),
                    point => new Point((int)Math.Round(point.X), (int)Math.Round(point.Y)));
                Cv2.Polylines(frame, new Point[][] { vertices }, true, Scalar.Red, 2);

                // Ensure the bounding rectangle is within the frame
                boundingRect = Rect.Intersect(boundingRect, new Rect(0, 0, frame.Width, frame.Height));

                // If the bounding rectangle is empty after the intersection, skip this box
                if (boundingRect.Width <= 0 || boundingRect.Height <= 0)
                    continue;

                boundingBoxes.Add(boundingRect);
            }
        }

        return boundingBoxes;
    }
}