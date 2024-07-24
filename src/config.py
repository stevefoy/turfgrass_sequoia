# config.py

BASE_INPUT_FOLDER = r'D:\datasets\Parrot\zendo_release\raw' # "/media/freddy/vault/datasets/Parrot/zendo_release/raw"
#BASE_PROCESSED_FOLDER = r'D:\datasets\Parrot\zendo_release\processed' # "/media/freddy/vault/datasets/Parrot/zendo_release/processed"
# BASE_PROCESSED_FOLDER = r'D:\datasets\Processed'
#BASE_PROCESSED_FOLDER = r'C:\Users\stevf\OneDrive\Documents\Projects\Github_IMVIP2024\Processed'
BASE_PROCESSED_FOLDER = r'C:\Users\stevf\OneDrive\Documents\Projects\Github_IMVIP2024\Processed_rectify'
BASE_PROCESSED_FOLDER_ALIGN = r'C:\Users\stevf\OneDrive\Documents\Projects\Github_IMVIP2024\Processed_align'
# List of folders
FOLDER_CAPTURES = [
    'ATU_01_MAY_2024', 'ATU_12_July_2023','ATU_24_APRIL_2024',
    'ATU_05_MAR_2024', 'ATU_14_MAY_2024', 'ATU_30_JAN_2024',
    'ATU_08_MAY_2024', 'ATU_19_FEB_2024', 'ATU_09_JUNE_2023',
    'ATU_20_MAR_2024', 'ATU_21_MAY_2024',
]

# List of folders
Folder_Details = {
    'ATU_01_MAY_2024':   ['0007', '0008', '0009', '0010', '0011'],
    'ATU_12_July_2023':  ['0110', '0111', '0113'],
    'ATU_24_APRIL_2024': ['0005'],
    'ATU_05_MAR_2024':   ['0128'],
    'ATU_14_MAY_2024':   ['0006'],
    'ATU_30_JAN_2024':   ['0118'],
    'ATU_08_MAY_2024':   ['0007'],
    'ATU_19_FEB_2024':   ['0123'],
    'ATU_09_JUNE_2023':  ['0094', '0095'],
    'ATU_20_MAR_2024':   ['0001', '0002'],
    'ATU_21_MAY_2024':   ['0016'],
}

CALILBRATION_NIR2RGB_JSON = r"D:\datasets\Parrot\zendo_release\raw\Calibration\align_NIR2RGB.json"
CALILBRATION_RED2NIR_JSON = r"D:\datasets\Parrot\zendo_release\raw\Calibration\align_RED2NIR.json"
