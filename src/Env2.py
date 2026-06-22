class Env2(Env):
    def __init__(self, post_coordinates=(0, 7), pos_goal=(11, 7)):
        super().__init__(post_coordinates, pos_goal)

    def init_episode(self):
        super().init_episode()

        walls = [(5,0),(5,1),(5,2)
        ]

        for wx, wy in walls:
            self._state[wx][wy] = 8

    def update(self, action):
        dif_x, dif_y = action
        coordinate_x, coordinate_y = self._coordinates

        r = 0
        term = False
        self._state[coordinate_x][coordinate_y] = 0

        next_x = self._limit_pos(coordinate_x + dif_x, 'width')
        next_y = self._limit_pos(coordinate_y + dif_y, 'height')

        if self._state[next_x][next_y] == 8:
            next_x, next_y = coordinate_x, coordinate_y
            r = -1

        elif self._state[next_x][next_y] == 9:
            r = 1
            term = True

        elif self._state[next_x][next_y] == 5:
            r = -1000
            next_x, next_y = self._s
        else:
            r = -1

        self._coordinates = (next_x, next_y)
        self._state[next_x][next_y] = 1

        return r, (next_x, next_y), term