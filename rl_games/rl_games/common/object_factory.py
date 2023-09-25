class ObjectFactory:
    def __init__(self):
        self._builders = {}

    def register_builder(self, name, builder):
        self._builders[name] = builder

    def set_builders(self, builders):
        self._builders = builders
        
    def create(self, name, **kwargs):
        builder = self._builders.get(name)
        if not builder:
            raise ValueError(name)
        return builder(**kwargs)