import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)


class Viewer(object):
    def __init__(self, start_time, patient_name, figsize=None):
        self.start_time = start_time
        self.patient_name = patient_name
        self.fig, self.axes, self.lines, self.ax2, self.ax1 = self.initialize()
        self.gird()
        self.update()

    def initialize(self):
        plt.ion()
        fig, axes = plt.subplots(4, figsize=(10, 8))

        axes[0].set_ylabel("BG (mg/dL)")
        axes[1].set_ylabel("CHO (g/min)")
        axes[2].set_ylabel("Insulin (U/min)")

        # 为COB坐标轴创建双坐标轴
        ax1 = axes[1].twinx()
        ax1.set_ylabel("COB (g)")

        # 为第三个子图创建双坐标轴
        ax2 = axes[2].twinx()
        ax2.set_ylabel("IOB (U)")

        axes[3].set_ylabel("Risk Index")

        (lineBG,) = axes[0].plot([], [], label="BG")
        (lineCGM,) = axes[0].plot([], [], label="CGM")
        (lineCHO,) = axes[1].plot([], [], label="CHO")
        (lineCOB,) = ax1.plot([], [], label="COB", linestyle="--", color="darkorange")
        (lineIns,) = axes[2].plot([], [], label="Insulin")
        (lineIOB,) = ax2.plot([], [], label="IOB", linestyle="--", color="darkorange")
        (lineLBGI,) = axes[3].plot([], [], label="Hypo Risk")
        (lineHBGI,) = axes[3].plot([], [], label="Hyper Risk")
        (lineRI,) = axes[3].plot([], [], label="Risk Index")

        lines = [
            lineBG,
            lineCGM,
            lineCHO,
            lineIns,
            lineLBGI,
            lineHBGI,
            lineRI,
            lineIOB,
            lineCOB,
        ]

        axes[0].set_ylim([70, 180])
        axes[1].set_ylim([0, 30])
        axes[2].set_ylim([-0.5, 1.0])  # 调整胰岛素坐标范围
        ax1.set_ylim([0, 6])  # 设置COB坐标范围
        ax2.set_ylim([-0.5, 3.0])  # 设置IOB坐标范围
        axes[3].set_ylim([0, 5])

        for ax in axes:
            ax.set_xlim([self.start_time, self.start_time + timedelta(hours=3)])
            ax.legend(loc="upper left")

        # 为双坐标轴添加图例
        ax1.legend(loc="upper right")
        ax2.legend(loc="upper right")

        # Plot zone patches
        axes[0].axhspan(70, 180, alpha=0.3, color="limegreen", lw=0)
        axes[0].axhspan(50, 70, alpha=0.3, color="red", lw=0)
        axes[0].axhspan(0, 50, alpha=0.3, color="darkred", lw=0)
        axes[0].axhspan(180, 250, alpha=0.3, color="red", lw=0)
        axes[0].axhspan(250, 1000, alpha=0.3, color="darkred", lw=0)

        axes[0].tick_params(labelbottom=False)
        axes[1].tick_params(labelbottom=False)
        axes[2].tick_params(labelbottom=False)
        axes[3].xaxis.set_minor_locator(mdates.AutoDateLocator())
        axes[3].xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M\n"))
        axes[3].xaxis.set_major_locator(mdates.DayLocator())
        axes[3].xaxis.set_major_formatter(mdates.DateFormatter("\n%b %d"))

        axes[0].set_title(self.patient_name)

        return fig, axes, lines, ax2, ax1

    def update(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def render(self, data):
        self.lines[0].set_xdata(data.index.values)
        self.lines[0].set_ydata(data["BG"].values)

        self.lines[1].set_xdata(data.index.values)
        self.lines[1].set_ydata(data["CGM"].values)

        self.axes[0].draw_artist(self.axes[0].patch)
        self.axes[0].draw_artist(self.lines[0])
        self.axes[0].draw_artist(self.lines[1])

        adjust_ylim(
            self.axes[0],
            min(min(data["BG"]), min(data["CGM"])),
            max(max(data["BG"]), max(data["CGM"])),
        )
        adjust_xlim(self.axes[0], data.index[-1])

        self.lines[2].set_xdata(data.index.values)
        self.lines[2].set_ydata(data["CHO"].values)
        # adjust_ylim(self.axes[1], min(data['CHO']), max(data['CHO']))

        self.lines[8].set_xdata(data.index.values)
        self.lines[8].set_ydata(data["COB"].values)
        # adjust_ylim_ax2(self.ax1, min(data['COB']), max(data['COB']))

        # 确保CHO和COB的0值对齐
        cho_min, cho_max = min(data["CHO"]), max(data["CHO"])
        cob_min, cob_max = min(data["COB"]), max(data["COB"])

        # 计算比例因子，使0值对齐
        cho_neg_range = abs(min(0, cho_min))
        cho_pos_range = max(0, cho_max)
        cho_y_min = -cho_neg_range * 1.1 if cho_neg_range > 0 else -0.1
        cho_y_max = cho_pos_range * 1.1 if cho_pos_range > 0 else 0.1

        # 计算COB轴的范围（保持0值对齐）
        cob_neg_range = abs(min(0, cob_min))
        cob_pos_range = max(0, cob_max)
        cob_y_min = -cob_neg_range * 1.1 if cob_neg_range > 0 else -0.1
        cob_y_max = cob_pos_range * 1.1 if cob_pos_range > 0 else 0.1

        # 分别设置两个轴的范围
        self.axes[1].set_ylim([cho_y_min, cho_y_max])
        self.ax1.set_ylim([cob_y_min, cob_y_max])

        self.axes[1].draw_artist(self.axes[1].patch)
        self.axes[1].draw_artist(self.lines[2])
        self.ax1.draw_artist(self.lines[8])

        adjust_xlim(self.axes[1], data.index[-1])

        # 更新胰岛素（主坐标轴）
        self.lines[3].set_xdata(data.index.values)
        self.lines[3].set_ydata(data["insulin"].values)

        # 更新IOB（次坐标轴）
        self.lines[7].set_xdata(data.index.values)
        self.lines[7].set_ydata(data["IOB"].values)

        # 计算比例因子，使两个轴的0点对齐且比例一致
        ins_max = max(data["insulin"]) if len(data["insulin"]) > -0.9 else 1.0
        iob_max = max(data["IOB"]) if len(data["IOB"]) > -0.9 else 5.0

        # 计算比例因子，保持两个轴的相对比例
        if ins_max > 0 and iob_max > 0:
            ratio = iob_max / ins_max
        else:
            ratio = 2.0  # 默认比例

        # 设置主坐标轴范围
        ins_y_max = ins_max * 1.2 if ins_max > 0 else 2.0
        self.axes[2].set_ylim([0, ins_y_max])

        # 设置次坐标轴范围，保持相同比例
        iob_y_max = ins_y_max * ratio
        self.ax2.set_ylim([0, iob_y_max])

        self.axes[2].draw_artist(self.axes[2].patch)
        self.axes[2].draw_artist(self.lines[3])
        self.ax2.draw_artist(self.lines[7])

        adjust_xlim(self.axes[2], data.index[-1])

        self.lines[4].set_xdata(data.index.values)
        self.lines[4].set_ydata(data["LBGI"].values)

        self.lines[5].set_xdata(data.index.values)
        self.lines[5].set_ydata(data["HBGI"].values)

        self.lines[6].set_xdata(data.index.values)
        self.lines[6].set_ydata(data["Risk"].values)

        self.axes[3].draw_artist(self.axes[3].patch)
        self.axes[3].draw_artist(self.lines[4])
        self.axes[3].draw_artist(self.lines[5])
        self.axes[3].draw_artist(self.lines[6])
        adjust_ylim(self.axes[3], min(data["Risk"]), max(data["Risk"]))
        adjust_xlim(self.axes[3], data.index[-1], xlabel=True)

        self.update()

    def gird(self):
        for ax in self.axes:
            ax.grid(linestyle="--")
        self.ax1.grid(linestyle=":", alpha=0.5)  # COB坐标轴也添加网格
        self.ax2.grid(linestyle=":", alpha=0.5)  # IOB坐标轴也添加网格
        self.update()

    def close(self):
        plt.close(self.fig)


def adjust_ylim(ax, ymin, ymax):
    ylim = ax.get_ylim()
    update = False

    if ymin < ylim[0]:
        y1 = ymin - 0.1 * abs(ymin)
        update = True
    else:
        y1 = ylim[0]

    if ymax > ylim[1]:
        y2 = ymax + 0.1 * abs(ymax)
        update = True
    else:
        y2 = ylim[1]

    if update:
        ax.set_ylim([y1, y2])
        for spine in ax.spines.values():
            ax.draw_artist(spine)
        ax.draw_artist(ax.yaxis)


def adjust_ylim_ax2(ax, ymin, ymax):
    """专门为双坐标轴的第二个轴调整Y轴范围"""
    ylim = ax.get_ylim()
    update = False

    if ymin < ylim[0]:
        y1 = ymin - 0.1 * abs(ymin)
        update = True
    else:
        y1 = ylim[0]

    if ymax > ylim[1]:
        y2 = ymax + 0.1 * abs(ymax)
        update = True
    else:
        y2 = ylim[1]

    if update:
        ax.set_ylim([y1, y2])
        for spine in ax.spines.values():
            ax.draw_artist(spine)
        ax.draw_artist(ax.yaxis)


def adjust_xlim(ax, timemax, xlabel=False):
    xlim = mdates.num2date(ax.get_xlim())
    update = False

    # remove timezone awareness to make them comparable
    timemax = timemax.replace(tzinfo=None)
    xlim[0] = xlim[0].replace(tzinfo=None)
    xlim[1] = xlim[1].replace(tzinfo=None)

    if timemax > xlim[1] - timedelta(minutes=30):
        xmax = xlim[1] + timedelta(hours=6)
        update = True

    if update:
        ax.set_xlim([xlim[0], xmax])
        for spine in ax.spines.values():
            ax.draw_artist(spine)
        ax.draw_artist(ax.xaxis)
        if xlabel:
            ax.xaxis.set_minor_locator(mdates.AutoDateLocator())
            ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M\n"))
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("\n%b %d"))
