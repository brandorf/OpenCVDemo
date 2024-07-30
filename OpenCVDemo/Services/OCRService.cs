using OpenCvSharp;
using OpenCvSharp.Dnn;
using OpenCvSharp.Text;

namespace OpenCVDemo.Services;

public class OCRService
{

    public OCRService()
    {
    }

    private static readonly string TessData = Path.Combine(AppContext.BaseDirectory, "Resources");

    public string Detect(Mat selectedDetectionFrame)
    {
        using (var image = CvDnn.BlobFromImage(selectedDetectionFrame))
        {
            var imageToOcr = image;
            if (selectedDetectionFrame.Type() != MatType.CV_8UC1)
            {
                imageToOcr = new Mat();
                Cv2.CvtColor(selectedDetectionFrame, imageToOcr, ColorConversionCodes.BGR2GRAY);
            }

            using (var tesseract = OCRTesseract.Create(TessData))

            {
                tesseract.Run(imageToOcr,
                    out var outputText, out var componentRects, out var componentTexts, out var componentConfidences);

                return outputText;

            }
        }
    }
    protected static Mat LoadImage(string fileName, ImreadModes modes = ImreadModes.Color) 
        => new(Path.Combine("_data", "image", fileName), modes);
}