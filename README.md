# GMTS Forecasting — Инструкция по запуску

Приложение для прогнозирования временных рядов на основе библиотеки Darts с веб-интерфейсом Streamlit.

---

## 🖥️ Требования

- **Windows 10/11**
- **Python 3.10 - 3.11** (рекомендуется 3.11)
- **Git** (опционально, для клонирования)
- **8+ GB RAM** (для моделей глубокого обучения)

---

## ⚡ Быстрая установка (Windows)

Для удобства доступен единый скрипт автоматической настройки:

1.  **setup.bat** — Создает виртуальное окружение, устанавливает PyTorch (CPU или GPU на выбор) и все необходимые библиотеки.
2.  **run.bat**   — Автоматически активирует окружение и запускает приложение.

> 💡 Просто дважды кликните на `setup.bat` при первом запуске, а затем используйте `run.bat`.

---

## 📦 Установка (пошагово)

### Шаг 1: Подготовка
Убедитесь, что у вас установлен **Python 3.10+**. При установке Python обязательно отметьте галочку **"Add Python to PATH"**.

### Шаг 2: Настройка окружения

```cmd
cd C:\путь\к\GMTS_v2
python -m venv venv
```

### Шаг 2: Активация окружения

**Для CMD:**
```cmd
venv\Scripts\activate.bat
```

**Для PowerShell:**
```powershell
venv\Scripts\Activate.ps1
```

> ⚠️ Если PowerShell выдаёт ошибку, выполните:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

После активации в начале строки появится `(venv)`.

### Шаг 3: Установка PyTorch

**Для GPU (NVIDIA CUDA 12.1):**
```cmd
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Для CPU (без GPU):**
```cmd
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Шаг 4: Установка зависимостей

```cmd
pip install -r requirements.txt
```

> ⏱️ Установка может занять 5-15 минут.

---

## 🚀 Запуск приложения

```cmd
python -m streamlit run streamlit_app.py
```

После запуска:
- В консоли появится ссылка: `http://localhost:3000`
- Браузер откроется автоматически

---

## 🔄 Повторный запуск

При каждом новом сеансе работы:

```cmd
cd C:\путь\к\GMTS_v2
venv\Scripts\activate.bat
streamlit run streamlit_app.py
```

---

## ❗ Решение проблем

| Проблема | Решение |
|----------|---------|
| `python не распознан` | Переустановите Python с галочкой "Add to PATH" |
| `pip не распознан` | Используйте `python -m pip install ...` |
| Ошибка при запуске Streamlit | Проверьте активацию venv: `(venv)` в начале строки |
| Ошибки импорта модулей | Повторите `pip install -r requirements.txt` |
| Нехватка памяти | Используйте модель `randomforest` вместо `nbeats`/`tftmodel` |

---

## 📁 Структура проекта

```
GMTS_v2/
├── streamlit_app.py      # Главное приложение
├── requirements.txt      # Зависимости Python
├── Data/                 # Данные для прогнозирования
└── utils/                # Вспомогательные модули
    ├── cluster_forecast.py
    ├── constants.py
    ├── parametrized_values.py
    └── ...
```