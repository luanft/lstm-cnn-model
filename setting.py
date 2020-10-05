"""
    Setting for stock price model
"""
import enum
# https://www.infobyip.com/detectmonitordpi.php
DPI = 144

# SIZE OF INPUT IMAGE TO CNN
IMG_W = 112
IMG_H = 112

# THE DIRECTORY FOR STORING CHART DATA
CHART_DIR = "C:\\Users\\a18822\\Desktop\\StockPrice\\CHART"


class ChartType(enum.Enum):
    BAR = "bar"
    LINE = "line"
    FLINE = "filled_line"
    CANDLESTICK = "candlestick"
    BAR_CANDLESTICK_FUSION = "bar_candlestick_fusion"
    BAR_LINE_FUSION = "bar_line_fusion"
    BAR_FLINE_FUSION = "bar_filled_line_fusion"

# tw_spydata_train.csv
TRAIN_SP500_DATA_FILE = "C:\\Users\\a18822\\Desktop\\StockPrice\\tw_spydata_train_mini.csv"
TEST_SP500_DATA_FILE = "C:\\Users\\a18822\\Desktop\\StockPrice\\tw_spydata_test_mini.csv"
VALIDATION_SP500_DATA_FILE = "C:\\Users\\a18822\\Desktop\\StockPrice\\tw_spydata_validation_mini.csv"

ALPHA = 0.2
BETA = 1
GAMMA = 0.2

# When generating the dataset, please set it to False
# When training or testing: set it to True
ENABLE_IMAGE_DECODE = True

WINDOW_SIZE = 30
PREDICT_SIZE = 5
BATCH_SIZE = 200
EPOCH = 2