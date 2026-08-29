import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import RectangleSelector

img = mpimg.imread('predictions/task1_train/colored_g_008.png')  # 改成你的案例

fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(img)
ax.set_title(f'拖拽画框，坐标会打印在终端  |  shape: {img.shape}')

def onselect(eclick, erelease):
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    print(f'框: 左上=({min(x1,x2)}, {min(y1,y2)})  宽={abs(x2-x1)}  高={abs(y2-y1)}')

selector = RectangleSelector(ax, onselect, useblit=True,
                             button=[1], minspanx=5, minspany=5, interactive=True)
plt.show()