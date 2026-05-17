using System.Windows;
using TsrApp.ViewModels;

namespace TsrApp
{
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
            DataContext = new MainViewModel();
        }
    }
}
