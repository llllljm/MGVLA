Atelectasis = "atelectasis atelectatic "
Cardiomegaly = "cardiomegaly  heart size cardiac enlargement cardiac size shadow contour " \
               "silhouette enlarged heart "
Consolidation = "consolidation consolidat "
Edema = "edema heart failure chf vascular congestion pulmonary congestion indistinctness vascular prominence"
PleuralE = "pleural fluid effusion "
EC = "enlarged cardiomediastinum mediastinumm cardiomediastinum contour mediastinal " \
     "configuration mmediastinal silhouette pericardial silhouette cardiac silhouette and " \
     "vascularity "
fracture = "fracture"
LungL = "lung lesion nodular density densities opacity opacities opacification nodule lump cavitary lesion carcinoma " \
        "neoplasm tumor "
LungO = "opaci decreased translucency increased density airspace air-space air space infiltrate infiltration " \
        "interstitial marking interstitial pattern interstitial lung reticular pattern reticular marking reticulation " \
        "parenchymal scarring peribronchial thickening wall thickening scar "
PleuralO = "pleural thickening fibrosis fibrothorax pleural scar pleural parenchymal scar pleuro-parenchymal scar " \
           "pleuro-pericardial scar "
Pneumonia = "pneumonia infection infectious process "
Pneumothorax = "pneumothorax pneumothoraces "
CATEGORIES_5 = {"Atelectasis": 0, "Cardiomegaly": 1, "Consolidation": 2,
                "Edema": 3, "Pleural Effusion": 4}
CATEGORIES = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Lesion",
              "Lung Opacity", "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
              "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"]
severity = "mild mildly minimal increased improved apperance improvement presistent moderate decreased small stable " \
           "large enlarged "
location = "left right upper lower mid bilateral bibasilar"
words = Atelectasis + Cardiomegaly + Consolidation + Edema + PleuralE + EC + fracture + LungL + LungO+ PleuralO + Pneumonia + Pneumothorax + severity + location
words_id = [5483, 24793, 10109, 2874, 2861, 3139, 9581, 3139, 2861, 19599, 7187, 8702, 8225, 2874, 6596, 6043, 1692, 3750, 2874, 3069, 6307, 3412, 10018, 3004, 10018, 30433, 27983, 1741, 3412, 15581, 2292, 2325, 2970, 3370, 8225, 10271, 11166, 12343, 1045, 10271, 11166, 7187, 6607, 9918, 2484, 1686, 5486, 3115, 8702, 6820, 8702, 3139, 8702, 1700, 17940, 4508, 2055, 13702, 4536, 12386, 3698, 10530, 7638, 8163, 11119, 10703, 25438, 4713, 5550, 4536, 4138, 8799, 2724, 6369, 1048, 2719, 5261, 14397, 14661, 1051, 2141, 3698, 19130, 3125, 17, 5441, 3125, 5441, 8665, 8442, 7612, 26891, 7612, 2574, 7612, 2572, 17794, 2574, 17794, 26891, 15916, 1710, 10750, 14377, 29472, 1821, 8539, 3628, 8539, 9063, 3752, 8539, 6119, 5037, 1728, 4390, 3752, 9063, 3752, 10750, 9063, 21372, 1036, 17, 10750, 9063, 21372, 1036, 17, 6820, 9063, 4130, 2585, 6057, 2488, 4809, 27241, 3024, 5706, 4727, 2141, 3254, 1922, 8727, 14661, 1035, 3902, 1968, 3179, 3540, 2719, 2538, 2899, 2878, 8225, 2279, 2354, 3802, 2425, 3192, 4026, 10232]

BERT_TYPE = 'microsoft/BiomedVLP-CXR-BERT-general'
VIT_TYPE = 'microsoft/swin-tiny-patch4-window7-224'
IMG_SIZE = 224
IMG_MEAN = .5862785803043838
IMG_STD = .27950088968644304

CHEXPERT_TASKS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Lung Opacity",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]
CHEXPERT_COMPETITION_TASKS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]
CHEXPERT_CLASS_PROMPTS = {
    "Atelectasis": {
        "severity": ["", "mild", "minimal"],
        "subtype": [
            "subsegmental atelectasis",
            "linear atelectasis",
            "trace atelectasis",
            "bibasilar atelectasis",
            "retrocardiac atelectasis",
            "bandlike atelectasis",
            "residual atelectasis",
        ],
        "location": [
            "at the mid lung zone",
            "at the upper lung zone",
            "at the right lung zone",
            "at the left lung zone",
            "at the lung bases",
            "at the right lung base",
            "at the left lung base",
            "at the bilateral lung bases",
            "at the left lower lobe",
            "at the right lower lobe",
        ],
    },
    "Cardiomegaly": {
        "severity": [""],
        "subtype": [
            "cardiac silhouette size is upper limits of normal",
            "cardiomegaly which is unchanged",
            "mildly prominent cardiac silhouette",
            "portable view of the chest demonstrates stable cardiomegaly",
            "portable view of the chest demonstrates mild cardiomegaly",
            "persistent severe cardiomegaly",
            "heart size is borderline enlarged",
            "cardiomegaly unchanged",
            "heart size is at the upper limits of normal",
            "redemonstration of cardiomegaly",
            "ap erect chest radiograph demonstrates the heart size is the upper limits of normal",
            "cardiac silhouette size is mildly enlarged",
            "mildly enlarged cardiac silhouette, likely left ventricular enlargement. other chambers are less prominent",
            "heart size remains at mildly enlarged",
            "persistent cardiomegaly with prominent upper lobe vessels",
        ],
        "location": [""],
    },
    "Consolidation": {
        "severity": ["", "increased", "improved", "apperance of"],
        "subtype": [
            "bilateral consolidation",
            "reticular consolidation",
            "retrocardiac consolidation",
            "patchy consolidation",
            "airspace consolidation",
            "partial consolidation",
        ],
        "location": [
            "at the lower lung zone",
            "at the upper lung zone",
            "at the left lower lobe",
            "at the right lower lobe",
            "at the left upper lobe",
            "at the right uppper lobe",
            "at the right lung base",
            "at the left lung base",
        ],
    },
    "Edema": {
        "severity": [
            "",
            "mild",
            "improvement in",
            "presistent",
            "moderate",
            "decreased",
        ],
        "subtype": [
            "pulmonary edema",
            "trace interstitial edema",
            "pulmonary interstitial edema",
        ],
        "location": [""],
    },
    "Pleural Effusion": {
        "severity": ["", "small", "stable", "large", "decreased", "increased"],
        "location": ["left", "right", "tiny"],
        "subtype": [
            "bilateral pleural effusion",
            "subpulmonic pleural effusion",
            "bilateral pleural effusion",
        ],
    },
}

COVID_TASKS = [
    'Normal',
    'COVID',
]
COVID_CLASS_PROMPTS = {
    'COVID': {
        'adjective': ['patchy','confluent'],
        'description': ['ground glass'],
        'subtype': ['opacity', 'consolidation'],
        'location': ['in peripheral', 'in mid', 'in lower'],
    }
}

RSNA_TASKS = [
    'Normal',
    'Pneumonia',
]
RSNA_CLASS_PROMPTS = {
    'Pneumonia': {
        'adjective': ['round', 'early', 'focal', 'multifocal', 'small', ''],
        'subtype': ['bacterial', 'viral', 'mycoplasma', ''],
        "location": [
            "at the mid lung zone",
            "at the upper lung zone",
            "at the right lung zone",
            "at the left lung zone",
            "at the lung bases",
            "at the right lung base",
            "at the left lung base",
            "at the bilateral lung bases",
            "at the left lower lobe",
            "at the right lower lobe",
            "at the left middle lobe",
            "at the right middle lobe",
            ""
        ]
    }
}