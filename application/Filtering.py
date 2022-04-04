class Filtering:
    def __init__(self, west=0, east=0, south=0, north=0, chart_manager=None):
        self.west, self.east, self.south, self.north = west, east, south, north
        self.chart_manager = chart_manager


    def set_chart_manager(self, chart_manager):
        self.chart_manager = chart_manager


    def apply(self, locations, data):

        fig, ax = None, None
        if self.chart_manager is not None:
            fig, ax, _ = self.chart_manager.create_figure((1,1), str(self))
            self.chart_manager.scatter_plot(ax, locations, 'r')
            self.chart_manager.plot_description(ax,x_label="Latitude", y_label="Longitude")

        indexes = [i for i in range(len(locations)) if
                   locations[i, 1] >= self.west and (locations[i, 1] <= self.east) and (
                           locations[i, 0] >= self.south) and locations[i, 0] <= self.north]


        locations = locations[indexes]

        if self.chart_manager is not None:
            self.chart_manager.scatter_plot(ax, locations, 'g')
        # for i,_ in enumerate(data):
        data = data[indexes]

        return locations, data


    def __str__(self):
        return "Filtering"