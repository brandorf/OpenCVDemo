using Microsoft.Maui.Controls.Shapes;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System.Text;
using OpenCVDemo.ViewModels;
using static OpenCvSharp.FileStorage;
using static System.Formats.Asn1.AsnWriter;
using Rect = OpenCvSharp.Rect;
using Size = OpenCvSharp.Size;

namespace OpenCVDemo.Services;

public interface IVideoProcessingService
{
    Task ProcessVideo(string videoFilePath);
    Detection ProcessSingleFrame(ImageSource image, float? confidenceOverride = null, Scalar? colorOverride = null);
    List<Detection> Detections { get; }
    decimal ProgressPercent { get; }

    decimal Fps { get; }
    int CurrentFrame { get; set; }
    int LastFrame { get; set; }
    TimeSpan FrameTime { get; set; }

    event Action ProgressChanged;
    event Action<Detection>? DetectionsChanged;
}