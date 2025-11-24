class Antenna:
    """
    Class Atenna
    """
    def __init__(self, antenna_id, longitude, latitude):
        self.id = antenna_id
        self.longitud = longitude
        self.latitud = latitude
        # Simulamos conexión a hardware
        print(f"  [Hardware] Antena {self.id} inicializada en {self.latitud}, {self.longitud}")

    def __repr__(self):
        return f"Antena(id={self.id})"
