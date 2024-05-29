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
    List<Detection> Detections { get; }
    decimal ProgressPercent { get; }

    event Action ProgressChanged;
    event Action DetectionsChanged;
}

public class VideoProcessingService : IVideoProcessingService
{
    private readonly ICVService _openCvService;
    private readonly decimal _progressPercent;


    public VideoProcessingService(ICVService openCvService)
    {
        _openCvService = openCvService;
        _openCvService.ProgressChanged += OnProgressChanged;
        _openCvService.DetectionsCHanged += OnDetectionsChanged;
    }

    private void OnDetectionsChanged()
    {
        DetectionsChanged?.Invoke();
    }

    private void OnProgressChanged()
    {
        ProgressChanged?.Invoke();
    }

    public async Task ProcessVideo(string videoFilePath)
    {
        await _openCvService.ProcessVideo(videoFilePath);
    }

    public List<Detection> Detections
    {
        get => _openCvService.Detections;
    }

    public decimal ProgressPercent => _openCvService.ProgressPercent;
    
    public event Action? ProgressChanged;
    public event Action? DetectionsChanged;

    private string GetTextRegionsDebugInfo(List<Mat> textRegions)
    {
        var sb = new StringBuilder();

        for (int i = 0; i < textRegions.Count; i++)
        {
            var textRegion = textRegions[i];
            sb.AppendLine($"Text region {i + 1}: {textRegion.Rows} rows, {textRegion.Cols} cols");
        }

        return sb.ToString();
    }
}