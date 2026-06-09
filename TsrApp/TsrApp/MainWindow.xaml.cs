using System.Windows;
using TsrApp.ViewModels;

namespace TsrApp
{
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
            var vm = new MainViewModel();
            DataContext = vm;
            Closed += (_, _) => vm.Dispose();
        }
    }
}
