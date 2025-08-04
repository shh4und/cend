from typing import List, Tuple

class SWCFile:
    def __init__(self, filename: str):
        self.filename: str = filename
        self.data: List[Tuple[float, int, float, float, float, float, int]] = []

    def add_point(
        self,
        identity: int,
        structure_type: int,
        x: float,
        y: float,
        z: float,
        radius: float,
        parent_identity: int
    ) -> None:
        if identity <= 0 or parent_identity >= identity or (parent_identity != -1 and parent_identity <= 0):
            print(f"identity value -> {identity}, parent value -> {parent_identity}")
            raise ValueError("Invalid identity or parent_identity values")
        self.data.append((identity, structure_type, x, y, z, radius, parent_identity))

    def write_file(self) -> bool:
        self.data.sort()
        with open(self.filename, 'w') as file:
            file.write("# SWC test pilot\n")
            for point in self.data:
                line = " ".join(map(str, point)) + "\n"
                file.write(line)
        print(">> SWC saved at:", self.filename)
        return True
