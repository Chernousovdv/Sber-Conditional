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
    "Кокосовое масло ": "Oils & Meals",
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
    "Какао-бобы": "Stimuli Crops",
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
    "Этилен, CFR Китай": "bump",

    # macros - do not create specific category for them due to different nature of data
    "Инфляция - Рост индекса цен производителей (RUB, eop PPI),": "Инфляция - Рост индекса цен производителей (RUB, eop PPI),",
    "Инфляция - Рост индекса потребительских цен в США, в долларах США (USD, eop CPI),": "Инфляция - Рост индекса потребительских цен в США, в долларах США (USD, eop CPI),",
    "Инфляция - Рост индекса цен производителей в США, в долларах США (USD, eop PPI),": "Инфляция - Рост индекса цен производителей в США, в долларах США (USD, eop PPI),",
    "Ключевая ставка, годовых": "Ключевая ставка, годовых",
    "Инфляция, г/г": "Инфляция, г/г",
    "USDRUB": "USDRUB"
}

## Best predictors gained for each cluster by experiments; could be changed
BEST_PREDICTORS_FOR_INDEX = {
    'Fish': ['USDRUB'],
    'Grains & Seeds': ['USDRUB'],
    'Livestock & Meat': ['USDRUB'],
    'Oils & Meals': ['USDRUB'],
    'Processed Meat': ['USDRUB'],
    'Stimuli Crops': ['USDRUB'],
    'Sugar & Flour': ['USDRUB'],
    'Vegetables': ['USDRUB'],
    'dip': ['USDRUB'],
    'bump': ['USDRUB'],
    "Инфляция - Рост индекса цен производителей (RUB, eop PPI),": ['USDRUB'],
    "Инфляция - Рост индекса потребительских цен в США, в долларах США (USD, eop CPI),": ['USDRUB'],
    "Инфляция - Рост индекса цен производителей в США, в долларах США (USD, eop PPI),": ['USDRUB'],
    "Ключевая ставка, годовых": ['USDRUB'],
    "Инфляция, г/г": ['USDRUB'],
    "USDRUB": None
}

CAT_MAP_ENCODING = {
 'Sugar & Flour': 0,
 'Vegetables': 1,
 'Oils & Meals': 2,
 'Fish': 3,
 'Livestock & Meat': 4,
 'Stimuli Crops': 5,
 'Processed Meat': 6,
 'Grains & Seeds': 7,
 'dip': 8,
 'bump': 9,
 'USDRUB': 10
}
