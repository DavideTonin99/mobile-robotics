class Dataset:

    def __init__(self, name, data):
        """
        **************************************************
        * Class Dataset                                  *
        **************************************************
        :param name: name of the dataset
        :param data: data of the dataset -> [{name: TimeSeries}]
        """
        self.name = name
        self.data = data