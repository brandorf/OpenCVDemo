using Microsoft.Extensions.Logging;
using OpenCVDemo.Services;
using OpenCVDemo.ViewModels;

namespace OpenCVDemo
{
    public static class MauiProgram
    {
        public static MauiApp CreateMauiApp()
        {
            var builder = MauiApp.CreateBuilder();
            builder
                .UseMauiApp<App>()
                .ConfigureFonts(fonts =>
                {
                    fonts.AddFont("OpenSans-Regular.ttf", "OpenSansRegular");
                    fonts.AddFont("OpenSans-Semibold.ttf", "OpenSansSemibold");
                });

            // Register your services here
            builder.Services.AddSingleton<IVideoProcessingService, VideoProcessingService>();
            builder.Services.AddSingleton<ICVService, OpenCvService>();
            builder.Services.AddSingleton<VideoProcessingViewModel>();
            builder.Services.AddTransient<VideoProcessingPage>();

#if DEBUG
            builder.Logging.AddDebug();
#endif

            return builder.Build();
        }
    }
}
