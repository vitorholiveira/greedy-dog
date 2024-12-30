class DogInstance:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.gpu_n = None
        self.gpu_vram = None
        self.prn_types_n = None
        self.prn_n = None
        self.prn = {}
        
        try:
            with open(filename, 'r') as file:
                self.gpu_n = int(file.readline())
                self.gpu_vram = int(file.readline())
                self.prn_types_n = int(file.readline())
                self.prn_n = int(file.readline())
                
                self.prn = {i: [] for i in range(self.prn_types_n)}
                
                for i in range(self.prn_n):
                    row = file.readline()
                    parsed_row = row.strip().split('\t')
                    type, vram = int(parsed_row[0]), int(parsed_row[1])
                    self.prn[type].append(vram)
                
            for key in self.prn:
                self.prn[key].sort()
        except:
            print(f"\ncan't read \"{filename}\"\n")

    def print_info(self) -> None:
        print("\n=============================")
        print("Dog Instance Info")
        print(self.filename)
        print("=============================")
        print(f"Number of GPU's: {self.gpu_n}")
        print(f"GPU's VRAM: {self.gpu_vram}")
        print(f"Number of PRN's: {self.prn_n}")
        print(f"Number of PRN's types: {self.prn_types_n}")
        print(f"Max PRN VRAM: {max(max(prn_vram) for prn_vram in self.prn.values())}")
        print(f"Min PRN VRAM: {min(min(prn_vram) for prn_vram in self.prn.values())}")
        print(f"Total PRN's VRAM: {sum(sum(prn_vram) for prn_vram in self.prn.values())}")
        print(f"Total GPU's VRAM: {self.gpu_n*self.gpu_vram}\n")

    def print_prns(self) -> None:
        print("\n=============================")
        print("PRN's VRAM by Type")
        print(self.filename)
        print("=============================")
        for type in self.prn:
            print(f"[{type+1}] : {self.prn[type]}")