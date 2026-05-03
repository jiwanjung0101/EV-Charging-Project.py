class ParkingLot:
    def __init__(self, lot_id, base_prices, congestion_rate=0.02, capacity=10):
        self.lot_id          = lot_id
        self.base_prices     = base_prices   # {t: price}
        self.congestion_rate = congestion_rate
        self.capacity        = capacity