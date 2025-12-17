from utils.clusterization_with_save import get_linear_clusters
from typing import Literal
import pandas as pd
import os
from utils.data_prepocessing import _ensure_monthly_index_and_align_exog


BASE_DIR = os.path.dirname(os.path.dirname(__file__))   # go up from utils/ to project root
DATA_DIR = os.path.join(BASE_DIR, "Data")

directory = f"{DATA_DIR}/"  # set directory path
d = {}
formed = {}

for i, entry in enumerate(os.scandir(directory)):  
    if entry.is_file() and entry.name.endswith('.xlsx'):  # check if it's a file
        d[entry.name] = pd.read_excel(f"{directory}{entry.name}")
    elif entry.is_file() and entry.name.endswith('.csv'):
        formed[entry.name] = pd.read_csv(f"{directory}{entry.name}")

global_macros = d["global_macro.xlsx"]
apk_nona = formed["apk_nona.csv"]

chemicals_nona = formed["chemicals_nona.csv"]
chemicals_nona["Date"] = pd.to_datetime(chemicals_nona["Date"])
chemicals_nona = chemicals_nona.set_index('Date')

global_macros = global_macros.drop(columns=["Инфляция - Рост индекса потребительских цен в США, в долларах США (USD, eop CPI),  .1"])

apk_nona["Date"] = apk_nona["Date"]
apk_nona["Date"] = pd.to_datetime(apk_nona["Date"])

global_macros = global_macros.iloc[3:][::-1].reset_index(drop=True)
global_macros = global_macros.drop(columns=["Date"])
global_macros.index = apk_nona["Date"]

apk_nona = apk_nona.set_index('Date')


assert global_macros.shape[0] == apk_nona.shape[0]


apk_nona_en, global_macros_en = _ensure_monthly_index_and_align_exog(apk_nona, global_macros)
chemicals_nona_en, global_macros_en_chemicals = _ensure_monthly_index_and_align_exog(chemicals_nona[chemicals_nona.index<=global_macros.index.max()],
                                                                                     global_macros[global_macros.index>=chemicals_nona.index.min()])
global_macros_en.columns = list(map(lambda x: " ".join(x.split()), list(global_macros_en.columns)))  # убираем табы


USE_DEFAULT_CATEGORY_MAP: bool = False
DISTANCE_METHOD: Literal["correlation", "Engle-Granger", "Johansen"] = "correlation"
LINKAGE_METHOD: Literal["single", "complete", "average", "ward"] = "ward"
THRESHOLD_FOR_CLUSTER_VAL: float = 0.2
DF_COMBINED: pd.DataFrame = pd.concat([apk_nona_en.dropna(), chemicals_nona_en.dropna()], axis=1)

CATEGORY_MAP = {
    "Кукуруза": "Зерно и семена",
    "Пшеница 1-го класса": "Зерно и семена",
    "Пшеница 3-го класса": "Зерно и семена",
    "Пшеница 4-го класса": "Зерно и семена",
    "Пшеница 5-го класса": "Зерно и семена",
    "Пшеница 12,5% FOB Ново, $ т": "Зерно и семена",
    "Подсолнечник": "Зерно и семена",
    "Соя": "Зерно и семена",
    "Рапс, руб. т": "Зерно и семена",
    
    # Масла и шроты
    "Подсолнечное масло наливом (мировые цены)": "Масла и шроты",
    "Бутилированное подсолнечное масло (рафинированное)": "Масла и шроты",
    "Подсолнечное масло (наливом) не бутилированное, нерафинированное ": "Масла и шроты",
    "Подсолнечное масло (наливом) не бутилированное, не": "Масла и шроты",
    "Подсолнечный шрот ": "Масла и шроты",
    "Соевое масло": "Масла и шроты",
    "Соевый шрот": "Масла и шроты",
    "Рапсовое масло EU, $ т": "Масла и шроты",
    "Кокосовое масло ": "Масла и шроты",
    "Пальмовое масло ": "Масла и шроты",
    
    # Животноводство
    "Молоко сырое": "Животноводство и мясо",
    "Мясо птицы бройлеров в живом весе ": "Животноводство и мясо",
    "Мясо крупного рогатого скота в живом весе ": "Животноводство и мясо",
    "Свинина в живом весе": "Животноводство и мясо",
    "Баранина в живом весе": "Животноводство и мясо",
    "Баранина в убойном весе ": "Животноводство и мясо",
    "Яйцо товарное": "Животноводство и мясо",
    
    # Переработанное мясо
    "Колбасы сырокопченые": "Переработанное мясо",
    "Колбасы вареные": "Переработанное мясо",
    
    # Рыба
    "Минтай б г, Владивосток руб. кг": "Рыба",
    "Минтай б г, Китай C&F $ т": "Рыба",
    "Сельдь н р, Владивосток руб. кг": "Рыба",
    
    # Овощи
    "Огурцы тепличные, руб. кг": "Овощи",
    "Томаты тепличные, руб. кг": "Овощи",
    
    # Другое
    "Сахар (средняя цена по России)": "Сахар и мука",
    "Мука пшеничная, руб. т": "Сахар и мука",
    
    # Стимулирующие культуры
    "Какао-бобы": "Стимулирующие культуры",
    "Табак": "Стимулирующие культуры",
    
    # Химические продукты
    # Удобрения и агрохимикаты (бывший dip)
    "Карбамид (FOB Южный)": "Удобрения и агрохимикаты",
    "Моноаммонийфосфат, MAP (FOB Балтика)": "Удобрения и агрохимикаты",
    "Апатитовый концетрат (FOB Morocco)": "Удобрения и агрохимикаты",
    "Аммиак (FOB Черное море)": "Удобрения и агрохимикаты",
    "Аммиачная селитра (FOB Черное море)": "Удобрения и агрохимикаты",
    "Хлорид калия (CFR Ю-В Азия)": "Удобрения и агрохимикаты",
    
    # Органические химикаты (бывший bump)
    "Капролактам импортный контракт (Тайвань и Ю. Корея) CFR Азия": "Органические химикаты",
    "Метанол": "Органические химикаты",
    "Бензол, CFR Япония": "Органические химикаты",
    "Этилен, CFR Китай": "Органические химикаты",
    
    # Макропоказатели - оставляем как есть
    "Инфляция - Рост индекса цен производителей (RUB, eop PPI),": "Инфляция - Рост индекса цен производителей (RUB, eop PPI),",
    "Инфляция - Рост индекса потребительских цен в США, в долларах США (USD, eop CPI),": "Инфляция - Рост индекса потребительских цен в США, в долларах США (USD, eop CPI),",
    "Инфляция - Рост индекса цен производителей в США, в долларах США (USD, eop PPI),": "Инфляция - Рост индекса цен производителей в США, в долларах США (USD, eop PPI),",
    "Ключевая ставка, годовых": "Ключевая ставка, годовых",
    "Инфляция, г/г": "Инфляция, г/г",
    "USDRUB": "USDRUB"
} if USE_DEFAULT_CATEGORY_MAP else get_linear_clusters(DF_COMBINED, DISTANCE_METHOD, LINKAGE_METHOD, THRESHOLD_FOR_CLUSTER_VAL)
for key in global_macros_en:
    if key not in CATEGORY_MAP:
        CATEGORY_MAP[key] = key

CATEGORY_MAP['USDRUB'] = "USDRUB"

## Best predictors gained for each cluster by experiments; could be changed
BEST_PREDICTORS_FOR_INDEX = {
    'Рыба': ['USDRUB'],
    'Зерно и семена': ['USDRUB'],
    'Животноводство и мясо': ['USDRUB'],
    'Масла и шроты': ['USDRUB'],
    'Переработанное мясо': ['USDRUB'],
    'Стимулирующие культуры': ['USDRUB'],
    'Сахар и мука': ['USDRUB'],
    'Овощи': ['USDRUB'],
    'Удобрения и агрохимикаты': ['USDRUB'],
    'Органические химикаты': ['USDRUB'],
    "Инфляция - Рост индекса цен производителей (RUB, eop PPI),": ['USDRUB'],
    "Инфляция - Рост индекса потребительских цен в США, в долларах США (USD, eop CPI),": ['USDRUB'],
    "Инфляция - Рост индекса цен производителей в США, в долларах США (USD, eop PPI),": ['USDRUB'],
    "Ключевая ставка, годовых": ['USDRUB'],
    "Инфляция, г/г": ['USDRUB'],
    "USDRUB": ['USDRUB']
} if USE_DEFAULT_CATEGORY_MAP else {v: None for v in CATEGORY_MAP.values()}

# CAT_MAP_ENCODING = {
#  'Сахар и мука': 0,
#  'Овощи': 1,
#  'Масла и шроты': 2,
#  'Рыба': 3,
#  'Животноводство и мясо': 4,
#  'Стимулирующие культуры': 5,
#  'Переработанное мясо': 6,
#  'Зерно и семена': 7,
#  'Удобрения и агрохимикаты': 8,
#  'Органические химикаты': 9,
#  'USDRUB': 10
# }
