CATEGORY_MAP = {
    "Кукуруза": "Grains & Seeds",
    "Пшеница 1-го класса": "Grains & Seeds",
    "Пшеница 3-го класса": "Grains & Seeds",
    "Пшеница 4-го класса": "Grains & Seeds",
    "Пшеница 5-го класса": "Grains & Seeds",
    "Пшеница 12,5% FOB Ново, $ т": "Grains & Seeds",
    "Подсолнечник": "Grains & Seeds",
    "Соя": "Grains & Seeds",
    "Рапс, руб. т": "Grains & Seeds",
    # Oils & Meals
    "Подсолнечное масло наливом (мировые цены)": "Oils & Meals",
    "Бутилированное подсолнечное масло (рафинированное)": "Oils & Meals",
    "Подсолнечное масло (наливом) не бутилированное, нерафинированное ": "Oils & Meals",
    "Подсолнечное масло (наливом) не бутилированное, не": "Oils & Meals",
    "Подсолнечный шрот ": "Oils & Meals",
    "Соевое масло": "Oils & Meals",
    "Соевый шрот": "Oils & Meals",
    "Рапсовое масло EU, $ т": "Oils & Meals",
    "Кокосовое масло (USD) ": "Oils & Meals",
    "Кокосовое масло ": "Oils & Meals",
    "Пальмовое масло (USD)": "Oils & Meals",
    "Пальмовое масло ": "Oils & Meals",
    # Livestock
    "Молоко сырое": "Livestock & Meat",
    "Мясо птицы бройлеров в живом весе ": "Livestock & Meat",
    "Мясо крупного рогатого скота в живом весе ": "Livestock & Meat",
    "Свинина в живом весе": "Livestock & Meat",
    "Баранина в живом весе": "Livestock & Meat",
    "Баранина в убойном весе ": "Livestock & Meat",
    "Яйцо товарное": "Livestock & Meat",
    # Processed meat
    "Колбасы сырокопченые": "Processed Meat",
    "Колбасы вареные": "Processed Meat",
    # Fish
    "Минтай б г, Владивосток руб. кг": "Fish",
    "Минтай б г, Китай C&F $ т": "Fish",
    "Сельдь н р, Владивосток руб. кг": "Fish",
    # Vegetables
    "Огурцы тепличные, руб. кг": "Vegetables",
    "Томаты тепличные, руб. кг": "Vegetables",
    # Other
    "Сахар (средняя цена по России)": "Sugar & Flour",
    "Мука пшеничная, руб. т": "Sugar & Flour",
    # Stimuli crops
    "Какао-бобы (USD)": "Stimuli Crops",
    "Какао-бобы": "Stimuli Crops",
    "Табак (USD)": "Stimuli Crops",
    "Табак": "Stimuli Crops",

    # Chemicals
    # dip
    "Карбамид (FOB Южный)": "dip",
    "Моноаммонийфосфат, MAP (FOB Балтика)": "dip",
    "Апатитовый концетрат (FOB Morocco)": "dip",
    "Аммиак (FOB Черное море)": "dip",
    "Аммиачная селитра (FOB Черное море)": "dip",
    "Хлорид калия (CFR Ю-В Азия)": "dip",

    # bump
    "Капролактам импортный контракт (Тайвань и Ю. Корея) CFR Азия": "bump",
    "Метанол": "bump",
    "Бензол, CFR Япония": "bump",
    "Этилен, CFR Китай": "bump"   
}

## Best predictors gained for each cluster by experiments; could be changed
BEST_PREDICTORS_FOR_INDEX = {
    'Fish': ['Oils & Meals', 'Sugar & Flour'],
    'Grains & Seeds': ['Oils & Meals', 'Processed Meat'],
    'Livestock & Meat': ['Processed Meat', 'Vegetables'],
    'Oils & Meals': ['Grains & Seeds', 'Vegetables'],
    'Processed Meat': ['Oils & Meals', 'Sugar & Flour'],
    'Stimuli Crops': ['Sugar & Flour', 'Vegetables'],
    'Sugar & Flour': ['Livestock & Meat', 'Vegetables'],
    'Vegetables': ['Livestock & Meat', 'Sugar & Flour'],
    'dip': None,
    'bump': None
}
BEST_PREDICTORS_FOR_INDEX_CHEMICALS = {
    'dip': None,
    'bump': None
}

CAT_MAP_ENCODING = {
 'Sugar & Flour': 0,
 'Vegetables': 1,
 'Oils & Meals': 2,
 'Fish': 3,
 'Livestock & Meat': 4,
 'Stimuli Crops': 5,
 'Processed Meat': 6,
 'Grains & Seeds': 7
}

CATEGORY_MAP_CHEMICALS = {
    "Карбамид (FOB Южный)": "dip",
    "Моноаммонийфосфат, MAP (FOB Балтика)": "dip",
    "Апатитовый концетрат (FOB Morocco)": "dip",
    "Аммиак (FOB Черное море)": "dip",
    "Аммиачная селитра (FOB Черное море)": "dip",
    "Хлорид калия (CFR Ю-В Азия)": "dip",
    "Капролактам импортный контракт (Тайвань и Ю. Корея) CFR Азия": "bump",
    "Метанол": "bump",
    "Бензол, CFR Япония": "bump",
    "Этилен, CFR Китай": "bump"   
}

CATEGORY_MAP_CHEMICALS_ENCODING = {
 'dip': 0,
 'bump': 1
}