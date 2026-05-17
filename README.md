# TsrApp — распознавание дорожных знаков

Курсовая работа по дисциплине «Теория принятия решений». Полный пайплайн: обучение свёрточной нейросети на наборе GTSRB на Python и десктопное WPF-приложение на C#, которое классифицирует изображения дорожных знаков, выполняя инференс обученной модели через ONNX Runtime.

**Точность на тестовой выборке: 99.42%** (43 класса немецких дорожных знаков, GTSRB).

---

## Оглавление

- [Общее описание](#общее-описание)
- [Архитектура решения](#архитектура-решения)
- [Структура репозитория](#структура-репозитория)
- [Часть 1. Обучение модели на Python](#часть-1-обучение-модели-на-python)
  - [Датасет GTSRB](#датасет-gtsrb)
  - [Модель ResNet-18 и transfer learning](#модель-resnet-18-и-transfer-learning)
  - [Препроцессинг и аугментации](#препроцессинг-и-аугментации)
  - [Цикл обучения](#цикл-обучения)
  - [Оценка качества](#оценка-качества)
  - [Экспорт в ONNX](#экспорт-в-onnx)
  - [Вспомогательные скрипты](#вспомогательные-скрипты)
- [Часть 2. Приложение TsrApp на C# и WPF](#часть-2-приложение-tsrapp-на-c-и-wpf)
  - [Технологический стек](#технологический-стек)
  - [Архитектура MVVM](#архитектура-mvvm)
  - [Слой Services](#слой-services)
  - [Слой ViewModels](#слой-viewmodels)
  - [Слой View](#слой-view)
  - [Логирование предсказаний](#логирование-предсказаний)
- [Соответствие препроцессинга Python и C#](#соответствие-препроцессинга-python-и-c)
- [Запуск и сборка](#запуск-и-сборка)
  - [Обучение на Python](#обучение-на-python)
  - [Сборка приложения на C#](#сборка-приложения-на-c)
  - [Готовый портативный билд](#готовый-портативный-билд)
- [Авторы](#авторы)

---

## Общее описание

Проект решает задачу классификации изображений дорожных знаков из набора **GTSRB** (German Traffic Sign Recognition Benchmark, 43 класса). Состоит из двух самостоятельных частей, связанных через формат **ONNX**:

1. **Python-часть (`tsr_train/`)** — обучение и оценка свёрточной нейронной сети на PyTorch, экспорт обученных весов в переносимый формат ONNX.
2. **C#-часть (`TsrApp/`)** — настольное Windows-приложение на WPF, которое загружает картинку с диска, прогоняет её через ONNX-модель с помощью ONNX Runtime и отображает результат вместе с историей предсказаний.

Такое разделение типично для прикладных ML-проектов: тяжёлая разработка модели идёт на удобной для исследователя экосистеме (Python + PyTorch + CUDA), а инференс выполняется на целевой платформе без зависимости от Python-окружения.

---

## Архитектура решения

```
[ GTSRB dataset ]
        |
        v
[ Python: PyTorch ]  --train-->  [ runs/best.pt ]
        |                              |
        |                              v
        |                       [ export_onnx.py ]
        |                              |
        |                              v
        +-------------> [ exports/model.onnx, labels.json ]
                                       |
                                       v
                       [ C# WPF + ONNX Runtime: TsrApp ]
                                       |
                                       v
                              [ Изображение знака ]
                                       |
                                       v
                          [ Класс знака + уверенность ]
```

Граница между частями — артефакт `model.onnx` плюс файл сопоставления индексов и имён классов `labels.json`. C#-приложение не зависит от PyTorch и Python; всё, что ему нужно, — это два файла в каталоге `Assets/` и нативная библиотека ONNX Runtime.

---

## Структура репозитория

```
CNN_for_TSR/
├── README.md                       # этот файл
├── .gitignore                      # исключает .venv, data, runs, bin, obj, publish
│
├── tsr_train/                      # Python: обучение и экспорт модели
│   ├── requirements.txt            # точные версии пакетов
│   ├── src/                        # основной код
│   │   ├── dataset.py              # GTSRB, разбиение train/val, аугментации, нормализация
│   │   ├── model.py                # сборка ResNet-18 с заменой головы на 43 класса
│   │   ├── train.py                # цикл обучения с early stopping
│   │   ├── evaluate.py             # оценка на тестовой выборке
│   │   └── export_onnx.py          # конвертация PyTorch → ONNX
│   └── scripts/                    # вспомогательные утилиты
│       ├── download_gtsrb.py       # скачивание датасета
│       ├── check_env.py            # диагностика окружения и CUDA
│       ├── check_onnx.py           # проверка экспортированной ONNX-модели
│       ├── plot_curves.py          # графики loss / accuracy
│       ├── plot_classification.py  # confusion matrix и метрики
│       ├── reference_logits.py     # эталонный inference для проверки parity C# ↔ Python
│       └── sample_test_images.py   # случайная выборка картинок для демонстрации
│
└── TsrApp/                         # C#: WPF-приложение
    ├── TsrApp.slnx                 # solution
    ├── sample_images/              # 20 тестовых картинок из GTSRB
    └── TsrApp/                     # сам проект
        ├── TsrApp.csproj           # net10.0-windows, NuGet-зависимости
        ├── App.xaml, App.xaml.cs   # точка входа WPF
        ├── MainWindow.xaml(.cs)    # главное окно
        ├── Assets/
        │   ├── model.onnx          # обученная модель (43 МБ)
        │   └── labels.json         # id → имя класса
        ├── Services/               # бизнес-логика
        │   ├── ImagePreprocessor.cs    # ресайз и нормализация
        │   ├── ClassifierService.cs    # ONNX Runtime + Softmax + top-3
        │   ├── PredictionLogger.cs     # CSV-лог
        │   └── PredictionLogEntry.cs   # запись лога
        ├── ViewModels/
        │   └── MainViewModel.cs    # связывание UI и сервисов
        └── Converters/
            └── HexStringToColorConverter.cs  # hex → Brush для индикатора уверенности
```

Папки, исключённые из репозитория и пересоздаваемые из кода:

| Папка | Размер | Чем пересобирается |
|---|---|---|
| `tsr_train/.venv/` | ~5 ГБ | `python -m venv .venv` + `pip install -r requirements.txt` |
| `tsr_train/data/` | ~700 МБ | `python scripts/download_gtsrb.py` |
| `tsr_train/runs/` | ~40 МБ | `python -m src.train` |
| `tsr_train/exports/` | 43 МБ | `python -m src.export_onnx` (уже лежит в `TsrApp/Assets/`) |
| `TsrApp/.../bin/`, `obj/` | — | `dotnet build` |
| `TsrApp/publish/` | ~200 МБ | `dotnet publish -c Release -r win-x64 --self-contained true` |

---

## Часть 1. Обучение модели на Python

### Датасет GTSRB

**GTSRB** (German Traffic Sign Recognition Benchmark) — стандартный набор для задачи распознавания дорожных знаков:

- 43 класса (запрещающие, предупреждающие, предписывающие и т. д.);
- ~39 000 обучающих картинок и ~12 600 тестовых;
- размер изображений варьируется (от ~15×15 до ~250×250 пикселей);
- сильный дисбаланс классов: самые распространённые встречаются на порядок чаще редких.

Датасет загружается через `torchvision.datasets.GTSRB`. Скрипт `scripts/download_gtsrb.py` тянет архивы в `tsr_train/data/`.

Разбиение train/val выполняется через **`StratifiedShuffleSplit`** (`src/dataset.py`): 80% / 20% от обучающей части с сохранением долей классов. Тестовая часть GTSRB используется как есть, без перемешивания с трейном.

### Модель ResNet-18 и transfer learning

Базовая архитектура — **ResNet-18** из `torchvision.models`, предобученная на ImageNet (`ResNet18_Weights.IMAGENET1K_V1`). В `src/model.py` финальный полносвязный слой `model.fc` заменяется на `nn.Linear(in_features, 43)`, чтобы выход соответствовал числу классов GTSRB.

Подход называется **transfer learning**: низкоуровневые признаки (края, текстуры, простые формы), которые сеть выучила на миллионе картинок ImageNet, переиспользуются, а под конкретную задачу адаптируются только верхние слои. Это сильно ускоряет сходимость и улучшает качество по сравнению с обучением с нуля на небольшом GTSRB.

Все параметры размораживаются и обучаются (fine-tuning всей сети), а не только новая голова.

### Препроцессинг и аугментации

Все картинки приводятся к фиксированному размеру **224×224** (родное разрешение ResNet) и нормализуются по статистикам ImageNet (`src/dataset.py`):

```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
```

Используются два разных пайплайна:

- **`train_tf`** — для обучения. Добавляет регуляризацию через аугментации:
  - `RandomAffine(degrees=15, translate=(0.05, 0.05))` — небольшие повороты и сдвиги, имитируют разные углы съёмки;
  - `ColorJitter(brightness=0.25, contrast=0.25)` — изменение яркости и контраста, имитирует разное освещение;
  - `GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))` — лёгкое размытие, имитирует расфокус и плохие условия.
- **`eval_tf`** — для валидации, теста и инференса. Только ресайз + ToTensor + нормализация, без аугментаций (детерминированный пайплайн).

`TransformedSubset` в `src/dataset.py` — обёртка над `torch.utils.data.Subset`, позволяющая применять **разные** трансформации к train- и val-подвыборкам одного и того же базового датасета без глобальной мутации `base.transform`.

### Цикл обучения

`src/train.py` реализует стандартный цикл с ранней остановкой:

| Параметр | Значение |
|---|---|
| Оптимизатор | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Функция потерь | CrossEntropyLoss |
| Batch size | 32 |
| Max epochs | 20 |
| LR scheduler | ReduceLROnPlateau (factor 0.1, patience 2) |
| Early stopping | patience 4 по val_loss |
| Seed | 42 |
| Device | CUDA, если доступна, иначе CPU |

Лучший чекпойнт по `val_loss` сохраняется в `runs/best.pt`, история обучения — в `runs/history.json`. Скрипт `scripts/plot_curves.py` строит графики loss/accuracy по этой истории.

### Оценка качества

`src/evaluate.py` загружает `runs/best.pt` и считает метрики на тестовой выборке GTSRB. Финальный результат:

- **Test accuracy: 99.42%**
- top-3 accuracy ещё выше.

`scripts/plot_classification.py` строит матрицу ошибок и пер-классовые метрики (precision, recall, F1), что помогает увидеть, на каких классах модель ошибается чаще всего.

### Экспорт в ONNX

`src/export_onnx.py` конвертирует обученный PyTorch-чекпойнт в **ONNX** (Open Neural Network Exchange) — переносимый формат, который понимают многие рантаймы (включая ONNX Runtime для C#).

- Входной тензор: имя `input`, форма `[1, 3, 224, 224]`, тип `float32`.
- Выходной тензор: имя `logits`, форма `[1, 43]`, тип `float32`.

Параллельно создаётся `labels.json` — словарь `{ "0": "класс 0", …, "42": "класс 42" }` для отображения человекочитаемых имён.

Оба файла копируются в `TsrApp/TsrApp/Assets/` и подключаются в `.csproj` через `<None ... CopyToOutputDirectory="PreserveNewest" />`, чтобы попадать в выходную папку при сборке.

### Вспомогательные скрипты

- **`scripts/check_env.py`** — печатает версии PyTorch / CUDA / ONNX Runtime, проверяет доступность GPU.
- **`scripts/check_onnx.py`** — открывает экспортированную модель в ONNX Runtime, прогоняет случайный батч и сверяет выход с PyTorch (sanity check после экспорта).
- **`scripts/reference_logits.py`** — фиксирует эталонные значения логитов и промежуточных тензоров для одного конкретного изображения. Используется для проверки **parity** между Python-инференсом и C#-инференсом (см. ниже).
- **`scripts/sample_test_images.py`** — случайно выбирает 20 картинок из тестового набора, конвертирует `.ppm → .png` и кладёт в `TsrApp/sample_images/` для демонстрации работы приложения без необходимости таскать с собой весь датасет.

---

## Часть 2. Приложение TsrApp на C# и WPF

### Технологический стек

| Слой | Технология |
|---|---|
| Целевой фреймворк | .NET 10 (TFM `net10.0-windows`) |
| UI | WPF (Windows Presentation Foundation, XAML) |
| Архитектурный паттерн | MVVM |
| MVVM-helpers | `CommunityToolkit.Mvvm` 8.x (атрибуты `[ObservableProperty]`, `[RelayCommand]`) |
| Инференс модели | `Microsoft.ML.OnnxRuntime` 1.26.0 |
| Обработка изображений | `SixLabors.ImageSharp` 3.1.12 (зафиксирована, т.к. 4.x требует платную лицензию) |
| CSV для лога | `CsvHelper` 33.x с `InvariantCulture` |

### Архитектура MVVM

Приложение построено по классическому **Model-View-ViewModel**:

- **View** (`MainWindow.xaml`) — описывает UI декларативно через XAML. Знает только про привязки `{Binding ...}` к свойствам ViewModel.
- **ViewModel** (`MainViewModel.cs`) — хранит состояние UI (`ImagePath`, `ResultText`, `ConfidenceColor`, `HasResult`, `Top3Text`, `History`) и команды (`LoadImageCommand`, `ClearHistoryCommand`). Не знает про XAML и контролы напрямую.
- **Model** — слой Services и POCO `PredictionLogEntry`, `PredictionResult`.

Связь View ↔ ViewModel выполняется через **data binding** WPF: ViewModel вызывает `OnPropertyChanged` (это делает `[ObservableProperty]` под капотом), биндинги получают уведомление и обновляют контрол. Это позволяет не писать `textBlock.Text = "..."` в code-behind и держать UI декларативным.

### Слой Services

#### `ImagePreprocessor.cs`

Превращает картинку с диска в массив `float[3*224*224]`, идентичный тому, что подаётся в PyTorch-модель в Python-eval-режиме:

1. Открывает изображение в формате `Rgb24` через ImageSharp.
2. Ресайзит до 224×224 с алгоритмом `KnownResamplers.Triangle` (билинейная фильтрация).
3. Идёт построчно: для каждого пикселя `(R, G, B)` делает `(value/255 - mean) / std` и раскладывает в **CHW**-порядок (сначала весь канал R, потом G, потом B) — именно такой порядок ожидает экспортированная ONNX-модель.

#### `ClassifierService.cs`

Точка входа в инференс. Принимает путь к `model.onnx` и `labels.json`:

- В конструкторе создаёт `InferenceSession`, парсит `labels.json` (через `System.Text.Json`) в `Dictionary<int, string>`.
- Метод `Predict(string imagePath)`:
  1. Вызывает `ImagePreprocessor.LoadAndPreprocess`.
  2. Оборачивает массив в `DenseTensor<float>` формы `[1, 3, 224, 224]`, кормит сессии.
  3. Получает логиты `float[43]`, применяет численно-стабильный **Softmax** (`exp(x - max(x))`), считает argmax и top-3.
- Возвращает `PredictionResult(ClassId, ClassName, Confidence, Top3)` — иммутабельный record.
- Реализует `IDisposable`, освобождая `InferenceSession` (это нативный ресурс).

#### `PredictionLogger.cs`

Тонкая обёртка над `CsvHelper` для логирования в `predictions_log.csv` рядом с исполняемым файлом:

- `Append(entry)` — дописывает запись; если файл не существует, сначала пишет заголовок.
- `ReadAll()` — читает всю историю при старте приложения.
- `Clear()` — перезаписывает файл, оставляя только заголовок.

Все три операции взаимно эксклюзивны через `lock(_lock)`, конфигурация CSV использует `CultureInfo.InvariantCulture`, чтобы десятичный разделитель был точкой независимо от системной локали.

#### `PredictionLogEntry.cs`

Простой POCO: `Timestamp` (ISO-8601 UTC), `ImagePath`, `PredictedClassId`, `PredictedClassName`, `Confidence`.

### Слой ViewModels

`MainViewModel` собирает приложение в единое целое:

- В конструкторе создаёт `ClassifierService` и `PredictionLogger`, путь к моделям берёт от `AppDomain.CurrentDomain.BaseDirectory` (рядом с `.exe`), читает существующий лог в `History`, сортируя по убыванию времени.
- Команда `LoadImageCommand` (генерируется атрибутом `[RelayCommand]` для метода `LoadImage`):
  - открывает `OpenFileDialog` с фильтром `.png/.jpg/.jpeg/.bmp`;
  - вызывает `_classifier.Predict(...)`;
  - заполняет UI-свойства: `ImagePath`, `ResultText`, `ConfidenceColor` (зелёный/жёлтый/красный по порогам 0.9 / 0.7), `Top3Text`, `HasResult = true`;
  - создаёт `PredictionLogEntry` и пишет его в лог и в начало `History` (тот же экземпляр).
- Команда `ClearHistoryCommand`:
  - показывает `MessageBox` с подтверждением;
  - при согласии вызывает `_logger.Clear()`, очищает `History`, сбрасывает все UI-свойства.

### Слой View

`MainWindow.xaml` — Grid с тремя строками:

| Строка | Высота | Содержимое |
|---|---|---|
| Верхняя | Auto | Кнопки «Загрузить изображение» и «Очистить историю» слева; цветная плашка с результатом классификации и список top-3 справа |
| Средняя | `*` | Превью загруженной картинки (`Image`, `Stretch="Uniform"`) |
| Нижняя | 240 | `DataGrid` с историей: время, путь к файлу, id, имя класса, уверенность |

Связь View → ViewModel настраивается в code-behind: `MainWindow.xaml.cs` после `InitializeComponent()` присваивает `DataContext = new MainViewModel()`.

Цвет плашки с результатом приходит от ViewModel как hex-строка (`#2E7D32`, `#F9A825`, `#C62828`), а `HexStringToColorConverter` (`Converters/`) на лету конвертирует её в `SolidColorBrush` — это позволяет логику выбора цвета держать в ViewModel, а не в XAML.

### Логирование предсказаний

Каждое распознавание дописывается строкой в `predictions_log.csv` в той же папке, что и `TsrApp.exe`. CSV-формат позволяет открывать историю в Excel или обрабатывать pandas. Сессии накапливаются: при старте приложение читает существующий лог и показывает его в нижней таблице, отсортированный по времени.

---

## Соответствие препроцессинга Python и C#

Чтобы C#-инференс давал тот же ответ, что и Python-эталон, оба пайплайна должны выполнять **математически одинаковую** последовательность преобразований над одной и той же картинкой. Это нетривиально, потому что:

- PIL (Python) использует свой вариант билинейной интерполяции (`Image.BILINEAR`);
- ImageSharp (C#) использует свой (`KnownResamplers.Triangle`).

Оба алгоритма называются «билинейными», но отличаются весами на границах пикселей. На картинке 224×224 это даёт расхождение порядка ±1/255 на ~15% пикселей.

Скрипт `tsr_train/scripts/reference_logits.py` сохраняет эталонные значения для одного изображения (`expected_input.bin` и `expected_logits.bin`). C#-сторона может сравнить свой результат с этим эталоном с **мягкими** порогами:

- по входному тензору: `max |diff| < 5e-2`;
- по логитам: `max |diff| < 1e-1`.

Несмотря на эту разницу, **argmax и top-3 классы совпадают** — что и требуется для практической работы.

Подробнее: нормализация по ImageNet-статистикам идентична в обеих сторонах (одни и те же mean/std), порядок осей CHW идентичен, тип `float32` идентичен. Единственный источник расхождения — алгоритм ресайза, и он не влияет на итоговое предсказание.

---

## Запуск и сборка

### Обучение на Python

```bash
cd tsr_train

# Окружение
python -m venv .venv
.venv/Scripts/activate          # Windows
pip install -r requirements.txt

# Датасет (~700 МБ)
python scripts/download_gtsrb.py

# Обучение (CUDA ускоряет на порядок)
python -m src.train

# Оценка на тесте
python -m src.evaluate

# Экспорт в ONNX
python -m src.export_onnx
# → tsr_train/exports/model.onnx, labels.json
```

### Сборка приложения на C#

Требования: **.NET 10 SDK** для Windows.

```bash
cd TsrApp

# Восстановление пакетов и сборка
dotnet build

# Запуск из консоли
dotnet run --project TsrApp/TsrApp.csproj
```

Solution-файл `TsrApp.slnx` открывается в Visual Studio 2022/2024.

### Готовый портативный билд

Самодостаточный билд под Windows x64 (без необходимости устанавливать .NET):

```bash
cd TsrApp
dotnet publish TsrApp/TsrApp.csproj -c Release -r win-x64 --self-contained true -o publish
```

Результат — папка `publish/` (~200 МБ), внутри `TsrApp.exe` и весь рантайм. Папку можно скопировать на любой Windows-компьютер и запускать двойным кликом.

В `publish/README.txt` лежит краткая пользовательская инструкция.

---

## Авторы

**Дмитриев А.А.**, **Стерхов С.Л.**
Группа Б24-171-1
ИжГТУ им. М.Т. Калашникова, 2026 г.
