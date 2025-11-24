from antenna import Antenna

class Interferometer:
    """
    Clase interferometer
    """
    def __init__(self, metadata):
        print(f"Iniciando Interferómetro con {len(metadata)} antenas...")
        self.antennas = []
        self.is_open = True
        
        # Load antennas
        for data in metadata:
            new_antenna = Antenna(
                antenna_id=data['id'],
                longitude=data['longitude'],
                latitude=data['latitude']
            )
            self.antennas.append(new_antenna)

    def reserve_calibration(self):  # Antes: reservar_calibracion
        """Logs the calibration reservation to the console."""
        if not self.is_open:
            raise RuntimeError("Interferometer is closed.")
        print(f"-> Reserving calibration for array with {len(self.antennas)} antennas.")

    def close(self):
        """
        Paso 2: Método explícito para liberar recursos.
        """
        if self.is_open:
            print("Cerrando interferómetro y liberando recursos...")
            self.antennas.clear() 
            self.is_open = False
        else:
            print("El interferómetro ya estaba cerrado.")

    def __del__(self):
        """
        Clears antennas from self.antennas
        """
        if hasattr(self, 'abierto') and self.is_open:
            print("¡Advertencia! Limpieza vía Garbage Collector (__del__)")
            self.close()
