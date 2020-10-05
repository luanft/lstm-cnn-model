import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mplfinance.original_flavor import candlestick2_ohlc
from util import convert_to_list, decode_img
from setting import DPI, IMG_H, IMG_W, CHART_DIR, ChartType

def init_chart():
    """Init the directory folder for charts"""
    dataset_type = ['train', 'validation', 'test']
    for ds_type in dataset_type:
        for ctype in ChartType:
            path_to_chart = os.path.join(CHART_DIR, ds_type, ctype.value)
            if not os.path.exists(path_to_chart):
                os.makedirs(path_to_chart, exist_ok=True)

def format_and_save_chart(path_to_image: str, fig_obj: Figure, *ax_objs: list):
    """Format chart for CNN input"""
    fig_obj.set_frameon(False)
    fig_obj.set_dpi(DPI)
    fig_obj.set_size_inches((IMG_W/DPI, IMG_H/DPI))
    for ax in ax_objs:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig_obj.tight_layout(pad=0.001)
    fig_obj.savefig(path_to_image)
    plt.close('all')
    plt.cla()
    plt.clf()

def create_bar_chart(fname, frame_data, chart_dir=''):
    """Create the bar chart from window data"""
    path_to_image = os.path.join(chart_dir, ChartType.BAR.value, "%s.png" % fname)
    if not os.path.exists(path_to_image):
        fig_obj, ax_obj = plt.subplots()
        time_series = convert_to_list(frame_data['Time'])
        closed_prices = convert_to_list(frame_data['Trade Volume'])
        ax_obj.bar(x=time_series, height=closed_prices, width=0.8, align='center')
        format_and_save_chart(path_to_image, fig_obj, ax_obj)
    return decode_img(path_to_image)

def create_line_chart(fname, frame_data, chart_dir=''):
    """Create the bar chart from window data"""
    path_to_image = os.path.join(chart_dir, ChartType.LINE.value, "%s.png" % fname)
    if not os.path.exists(path_to_image):
        fig_obj, ax_obj = plt.subplots(nrows=1)
        time_series = convert_to_list(frame_data['Time'])
        high_prices = convert_to_list(frame_data['Trade High'])
        low_prices = convert_to_list(frame_data['Trade Low'])
        ax_obj.plot(time_series, high_prices, color='green')
        ax_obj.plot(time_series, low_prices, color='red')
        format_and_save_chart(path_to_image, fig_obj, ax_obj)
    return decode_img(path_to_image)


def create_filled_line_chart(fname, frame_data, chart_dir=''):
    """Create the F line chart from window data"""
    path_to_image = os.path.join(chart_dir, ChartType.FLINE.value, "%s.png" % fname)
    if not os.path.exists(path_to_image):
        fig_obj, ax_obj = plt.subplots(nrows=1)
        time_series = convert_to_list(frame_data['Time'])
        high_prices = convert_to_list(frame_data['Trade High'])
        low_prices = convert_to_list(frame_data['Trade Low'])
        mean_prices = ((np.array(high_prices) + np.array(low_prices)) / 2).tolist()
        ax_obj.plot(time_series, high_prices, color='green', linewidth=0.1)
        ax_obj.plot(time_series, low_prices, color='red', linewidth=0.1)
        ax_obj.fill_between(time_series, high_prices, mean_prices, color='green')
        ax_obj.fill_between(time_series, mean_prices, low_prices, color='red')
        format_and_save_chart(path_to_image, fig_obj, ax_obj)
    return decode_img(path_to_image)

def create_candlestick_chart(fname, frame_data, chart_dir=''):
    """Create the candlestick chart from window data"""
    path_to_image = os.path.join(chart_dir, ChartType.CANDLESTICK.value, "%s.png" % fname)
    if not os.path.exists(path_to_image):
        fig_obj, ax_obj = plt.subplots()
        high_prices = convert_to_list(frame_data['Trade High'])
        low_prices = convert_to_list(frame_data['Trade Low'])
        open_prices = convert_to_list(frame_data['Trade Open'])
        closed_prices = convert_to_list(frame_data['Trade Close'])
        candlestick2_ohlc(
            ax_obj, opens=open_prices, highs=high_prices, lows=low_prices,
            closes=closed_prices, width=0.5, colorup='green', colordown='red', alpha=0.8
        )
        format_and_save_chart(path_to_image, fig_obj, ax_obj)

    return decode_img(path_to_image)

def draw_fusion_bar_chart(ax_obj, time_series, trading_volume):
    """Draw the bar char"""
    ax_bar_obj: Axes = ax_obj.twinx()
    ax_bar_obj.autoscale_view()
    ax_bar_obj.set_ylim((0, max(trading_volume) * 2))
    ax_bar_obj.bar(x=time_series, height=trading_volume, width=0.8, align='center', color="blue")
    return ax_bar_obj

def create_bar_candlestick_fusion_chart(fname, frame_data, chart_dir=''):
    """Create the bar candlestick fusion chart from window data"""
    path_to_image = os.path.join(chart_dir, ChartType.BAR_CANDLESTICK_FUSION.value, "%s.png" % fname)
    if not os.path.exists(path_to_image):
        fig_obj, ax_candle_obj = plt.subplots()
        time_series = convert_to_list(frame_data['Time'])
        trading_volume = convert_to_list(frame_data['Trade Volume'])
        high_prices = convert_to_list(frame_data['Trade High'])
        low_prices = convert_to_list(frame_data['Trade Low'])
        open_prices = convert_to_list(frame_data['Trade Open'])
        closed_prices = convert_to_list(frame_data['Trade Close'])
        ax_bar_obj = draw_fusion_bar_chart(ax_candle_obj, list(range(len(time_series))), trading_volume)
        candlestick2_ohlc(
            ax_candle_obj, opens=open_prices, highs=high_prices, lows=low_prices,
            closes=closed_prices, width=0.5, colorup='black', colordown='red', alpha=0.8
        )
        format_and_save_chart(path_to_image, fig_obj, ax_candle_obj, ax_bar_obj)
    return decode_img(path_to_image)

def create_bar_line_fusion_chart(fname, frame_data, chart_dir=''):
    """Create the bar line fusion chart from window data"""
    path_to_image = os.path.join(chart_dir, ChartType.BAR_LINE_FUSION.value, "%s.png" % fname)
    if not os.path.exists(path_to_image):
        fig_obj, ax_line_obj = plt.subplots()
        time_series = convert_to_list(frame_data['Time'])
        trading_volume = convert_to_list(frame_data['Trade Volume'])
        high_prices = convert_to_list(frame_data['Trade High'])
        low_prices = convert_to_list(frame_data['Trade Low'])
        transformed_time_series = list(range(len(time_series)))
        ax_bar_obj = draw_fusion_bar_chart(ax_line_obj, transformed_time_series, trading_volume)
        ax_line_obj.plot(transformed_time_series, high_prices, color='green')
        ax_line_obj.plot(transformed_time_series, low_prices, color='red')
        format_and_save_chart(path_to_image, fig_obj, ax_line_obj, ax_bar_obj)
    return decode_img(path_to_image)

def create_bar_filled_line_fusion_chart(fname, frame_data, chart_dir=''):
    """Create the bar filled line fusion chart from window data"""
    path_to_image = os.path.join(chart_dir, ChartType.BAR_FLINE_FUSION.value, "%s.png" % fname)
    if not os.path.exists(path_to_image):
        fig_obj, ax_fline_obj = plt.subplots()
        time_series = convert_to_list(frame_data['Time'])
        trading_volume = convert_to_list(frame_data['Trade Volume'])
        high_prices = convert_to_list(frame_data['Trade High'])
        low_prices = convert_to_list(frame_data['Trade Low'])
        mean_prices = ((np.array(high_prices) + np.array(low_prices)) / 2).tolist()
        transformed_time_series = list(range(len(time_series)))
        ax_bar_obj = draw_fusion_bar_chart(ax_fline_obj, transformed_time_series, trading_volume)
        ax_fline_obj.plot(transformed_time_series, high_prices, color='green', linewidth=0.1)
        ax_fline_obj.plot(transformed_time_series, low_prices, color='red', linewidth=0.1)
        ax_fline_obj.fill_between(transformed_time_series, high_prices, mean_prices, color='green')
        ax_fline_obj.fill_between(transformed_time_series, mean_prices, low_prices, color='red')
        format_and_save_chart(path_to_image, fig_obj, ax_fline_obj, ax_bar_obj)
    return decode_img(path_to_image)


init_chart()