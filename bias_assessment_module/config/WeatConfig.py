class WeatConfig():
    def __init__(self, config):
        self._config = config

    @property
    def name(self):
        return self._config["name"]

    @property
    def a(self):
        return self._config["a"]

    @property
    def b(self):
        return self._config["b"]

    @property
    def x(self):
        return self._config["x"]

    @property
    def y(self):
        return self._config["y"]

    def copy(self, updates=None):
        new_config = self._config.copy()
        if updates is not None:
            new_config.update(updates)
        return WeatConfig(new_config)
