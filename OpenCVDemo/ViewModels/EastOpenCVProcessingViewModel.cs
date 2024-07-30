using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using OpenCVDemo.Services;
using OpenCvSharp;
using OpenCvSharp.Internal.Vectors;
using Rect = OpenCvSharp.Rect;

namespace OpenCVDemo.ViewModels;

public class Detection
{
    public Detection()
    {
        this.Id = Guid.NewGuid();
    }

    public Guid Id { get; private set; }
    public Mat Frame { get; set; }
    public List<Rect> BoundingBoxes { get; set; }

    public ImageSource FrameImageSource => ImageSource.FromStream(() => new MemoryStream(MatToBytes(Frame)));

    private byte[] MatToBytes(Mat image)
    {
        Cv2.ImEncode(".png", image, out var vec);
        byte[] result = vec.ToArray();
        return result;
    }

    private Mat DrawBoundingBoxes()
    {
        foreach (var box in BoundingBoxes)
        {
            Cv2.Rectangle(Frame, box, Scalar.Green, 2);
        }

        return Frame;
    }

    public override string ToString()
    {
        return $"{Id}: Detections [{BoundingBoxes.Count}]";
    }
}

public class EastOpenCVProcessingViewModel : INotifyPropertyChanged
{
    private readonly IVideoProcessingService _videoProcessingService;
    private readonly OCRService _ocrService;
    private float _confidence;

    private ObservableCollection<Detection> _detections = new ObservableCollection<Detection>();

    private bool _isProcessing;
    private string _processingResult;
    private Detection _selectedDetection;
    private string _videoFilePath;

    public EastOpenCVProcessingViewModel(IVideoProcessingService videoProcessingService, OCRService ocrService)
    {
        _videoProcessingService = videoProcessingService;
        _ocrService = ocrService;
        _videoProcessingService.ProgressChanged += OnProgressChanged;
        _videoProcessingService.DetectionsChanged += OnDetectionsChanged;
        SelectFileCommand = new Command(async () => await SelectFile());
        ProcessVideoCommand = new Command(ProcessVideo, CanProcessVideo);
        
    }

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

    public Command ProcessVideoCommand { get; set; }

    public Command SelectFileCommand { get; set; }

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

    public ObservableCollection<Detection> Detections
    {
        get => _detections;
        set => _detections = value;
    }
    
    public string SelectedTextDetection { get; set; }

    public Detection SelectedDetection

    {
        get => _selectedDetection;
        set
        {
            if (Equals(value, _selectedDetection)) return;
            _selectedDetection = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(SelectedDetectionImage));
            Console.WriteLine($"Changed selected detection to {value}");

                var worker = new BackgroundWorker();

                worker.DoWork += (sender, args) => args.Result = _ocrService.Detect(_selectedDetection.Frame);
                worker.RunWorkerCompleted += (sender, args) =>
                {
                    if (args.Error != null)
                    {
                        // Handle the error
                    }
                    SelectedTextDetection = args.Result.ToString();
                    OnPropertyChanged(nameof(SelectedTextDetection));
                    IsProcessing = false;
                };

                worker.RunWorkerAsync();
            
            
        }
    }

    public ImageSource SelectedDetectionImage => SelectedDetection?.FrameImageSource;

    public decimal ProgressPercent => _videoProcessingService.ProgressPercent;
    public decimal FPS => _videoProcessingService.Fps;

    public string ProcessingMessage => $"Processing frame {_videoProcessingService.CurrentFrame} of {_videoProcessingService.LastFrame}";
    public TimeSpan EstimatedTimeRemaining
    {
        get
        {
            var framesRemaining = _videoProcessingService.LastFrame - _videoProcessingService.CurrentFrame;
            return TimeSpan.FromTicks((long)(framesRemaining * _videoProcessingService.FrameTime.Ticks));
        }
    }

    public event PropertyChangedEventHandler? PropertyChanged;

    private bool CanProcessVideo()
    {
        return !IsProcessing && !String.IsNullOrWhiteSpace(VideoFilePath);
    }

    public async void ProcessVideo()
    {
        IsProcessing = true;
        Detections.Clear();

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

    private void OnProgressChanged()
    {
        OnPropertyChanged(nameof(ProgressPercent));
        OnPropertyChanged(nameof(FPS));
        OnPropertyChanged(nameof(EstimatedTimeRemaining));
        OnPropertyChanged(nameof(ProcessingMessage));
    }

    private void OnDetectionsChanged(Detection newDetection)
    {
        // Update the Detections collection on the UI thread
        Device.InvokeOnMainThreadAsync(() =>
        {
            Detections.Add(newDetection);
        });
    }

    protected virtual void OnPropertyChanged([CallerMemberName] string? propertyName = null)
    {
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
    }
}