using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCVDemo.Services;
using OpenCVDemo.ViewModels;

namespace OpenCVDemo;

public partial class VideoProcessingPage : ContentPage
{
    public VideoProcessingPage(EastOpenCVProcessingViewModel vm)
        {
            InitializeComponent();
            BindingContext = vm;
        }

}