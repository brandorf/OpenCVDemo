using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using OpenCVDemo.Services;
using OpenCvSharp;
using Rect = OpenCvSharp.Rect;

namespace OpenCVDemo.ViewModels;

public class Detection
{
    public Mat Frame { get; set; }
    public List<Rect> BoundingBoxes { get; set; }
}

public class VideoProcessingViewModel : INotifyPropertyChanged
{
    private readonly IVideoProcessingService _videoProcessingService;
    private string _videoFilePath;
    private string _processingResult;

    public VideoProcessingViewModel(IVideoProcessingService videoProcessingService)
    {
        _videoProcessingService = videoProcessingService;
        _videoProcessingService.ProgressChanged += OnProgressChanged;
        _videoProcessingService.DetectionsChanged += OnDetectionsChanged;
        SelectFileCommand = new Command(async () => await SelectFile());
        ProcessVideoCommand = new Command(ProcessVideo, CanProcessVideo);
    }

    private bool _isProcessing;

    public bool IsProcessing
    {
        get => _isProcessing;
        set
        {
            if (_isProcessing != value)
            {
                _isProcessing = value;
                OnPropertyChanged();
                ProcessVideoCommand.ChangeCanExecute();
            }
        }
    }

    private bool CanProcessVideo()
    {
        return !IsProcessing && !String.IsNullOrWhiteSpace(VideoFilePath);
    }

    public Command ProcessVideoCommand { get; set; }

    public Command SelectFileCommand { get; set; }

    public async void ProcessVideo()
    {
        IsProcessing = true;

        var worker = new BackgroundWorker();

        worker.DoWork += (sender, args) => _videoProcessingService.ProcessVideo(VideoFilePath);
        worker.RunWorkerCompleted += (sender, args) =>
        {
            if (args.Error != null)
            {
                // Handle the error
            }

            IsProcessing = false;
        };

        worker.RunWorkerAsync();
    }

    private async Task SelectFile()
    {
        if (IsProcessing)
            return;

        var result = await FilePicker.PickAsync();
        if (result != null)
        {
            VideoFilePath = result.FullPath;
        }
    }

    public string VideoFilePath
    {
        get => _videoFilePath;
        set
        {
            if (value == _videoFilePath) return;
            _videoFilePath = value;
            OnPropertyChanged();
            ProcessVideoCommand.ChangeCanExecute();
        }
    }

    private ObservableCollection<Detection> _detections;

    public ObservableCollection<Detection> Detections
    {
        get => new ObservableCollection<Detection>(_videoProcessingService.Detections);
    }

    public Detection SelectedDetection { get; set; }

    private void OnProgressChanged()
    {
        OnPropertyChanged(nameof(ProgressPercent));
        OnPropertyChanged(nameof(FPS));
        OnPropertyChanged(nameof(EstimatedTimeRemaining));
    }

    private void OnDetectionsChanged()
    {
        OnPropertyChanged(nameof(Detections));
    }

    public decimal ProgressPercent => _videoProcessingService.ProgressPercent;
    public decimal FPS => _videoProcessingService.Fps;

    public TimeSpan EstimatedTimeRemaining
    {
        get
        {
            var framesRemaining = _videoProcessingService.LastFrame - _videoProcessingService.CurrentFrame;
            return TimeSpan.FromTicks((long)(framesRemaining * _videoProcessingService.FrameTime.Ticks));
        }
    }

    public event PropertyChangedEventHandler? PropertyChanged;

    protected virtual void OnPropertyChanged([CallerMemberName] string? propertyName = null)
    {
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
    }

    protected bool SetField<T>(ref T field, T value, [CallerMemberName] string? propertyName = null)
    {
        if (EqualityComparer<T>.Default.Equals(field, value)) return false;
        field = value;
        OnPropertyChanged(propertyName);
        return true;
    }
}