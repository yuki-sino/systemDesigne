import math
import random
import matplotlib.pyplot as plt
# 確率,報酬、分散

class RiskMap(object):
    def __init__(self):
        self._state = []

    # sources: 安全圏を作るための点集合
    def makeMap(self, sources = None):
        # 3,4
        random.seed(8)
        self._state = []
        probMap = [
            [
                (p := random.random(), -1 / p, (1-p)/p)
                for _ in range(12)
            ]
            for _ in range(8)
        ]


        for x in range(8):

            row = []

            for y in range(12):

                danger = 0

                # Gaussian危険源
                for sx, sy in sources:

                    d2 = (x - sx) ** 2 + (y - sy) ** 2

                    danger += math.exp(-d2 / 5)

                # 0~1に制限
                danger = max(0, min(danger, 1))

                if not(danger > 0.005 and danger < 0.08):
                    danger = 1
                else:
                    danger = 0
                row.append((danger))

            self._state.append(row)

        riskmap = [
            [
                (
                    1 if self._state[x][y] == 0 else probMap[x][y][0],
                    probMap[x][y][1] * self._state[x][y],
                    probMap[x][y][2] * self._state[x][y]
                )
                for y in range(12)
            ]
            for x in range(8)
        ]
        return riskmap
    
    # def showMap(self, riskmap):
            
    #     values = [[cell[2] for cell in row] for row in riskmap]
    #     plt.imshow(values, cmap="viridis", origin="upper", aspect="auto")

    #     for x in range(8):
    #         for y in range(12):
    #             if self._state[x][y] == 0:
    #                 plt.scatter(y, x, color="red", s=20)

    #     plt.colorbar(label="value")
    #     plt.show()

    def showMap(self, riskmap):
        width, height = 12, 8
        
        C = np.array([[riskmap[x][y][2] for y in range(width)] for x in range(height)])  # shape: (8, 12) = (height, width)

        rows = np.arange(width + 1)   # 13要素
        cols = np.arange(height + 1)  # 9要素
        X, Y = np.meshgrid(rows, cols)  # (9,13)

        plt.figure(figsize=(10, 5))
        plt.xlim(0, width)
        plt.ylim(height, 0)
        plt.pcolor(X, Y, C, cmap=plt.cm.viridis)
        plt.grid(True, which='both', linestyle='-', color='k')

        # 安全圏を赤点でオーバーレイ
        for x in range(8):
            for y in range(12):
                if self._state[x][y] == 0:
                    plt.scatter(y + 0.5, x + 0.5, color="red", s=40, zorder=5)

        # sourcesの位置
        for sx, sy in [(5, 5), (7, 7)]:
            plt.scatter(sy + 0.5, sx + 0.5, color="blue", s=120, marker='X', zorder=6)

        plt.title('RiskMap  black=safe zone(red dot)  white=danger  blue X=source')
        plt.tight_layout()
        plt.show()