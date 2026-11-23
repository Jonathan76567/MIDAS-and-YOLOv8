import json
import os

class CostCalculator:
    def __init__(self, price_list_path='data/price_list.json'):
        if not os.path.exists(price_list_path):
            raise FileNotFoundError(f"File database harga tidak ditemukan di {price_list_path}")
            
        with open(price_list_path, 'r') as f:
            self.db = json.load(f)
            
        self.alpha = self.db.get("bobot_formula", {}).get("alpha_luas", 0.5)
        self.beta = self.db.get("bobot_formula", {}).get("beta_kedalaman", 0.8)

        self.panel_mapping = {k: v for k, v in self.db.get("panel_mapping", {}).items()
                              if not k.startswith("_")}
        self.depth_classes = set(self.db.get("kelas_kedalaman", {}).get("kelas", []))

    def panel_names(self):
        return list(self.db.get("harga_base_panel", {}).keys())

    def calculate_cost(self, panel_name, mask_area_ratio, depth_deviation):
        """
        Menghitung harga dinamis menggunakan fungsi matematis:
        P_total = P_base * (1 + alpha * A_mask + beta * D_depth)
        
        Param:
        - panel_name (str): Nama panel (e.g. 'pintu_depan')
        - mask_area_ratio (float): Rasio luas mask kerusakan terhadap panel (0.0 - 1.0)
        - depth_deviation (float): Deviasi kedalaman penyok dari model kedalaman (0.0 - 1.0)
        """
        base_prices = self.db.get("harga_base_panel", {})
        
        if panel_name not in base_prices:
            return None, f"Panel '{panel_name}' tidak terdaftar di database harga dasar."

        p_base = base_prices[panel_name]
        
        # Pengunci nilai masukan agar berada di rentang 0.0 - 1.0
        a_mask = max(0.0, min(1.0, mask_area_ratio))
        d_depth = max(0.0, min(1.0, depth_deviation))
        
        # Perhitungan pengganda biaya tambahan
        multiplier = 1.0 + (self.alpha * a_mask) + (self.beta * d_depth)
        
        # Total harga dihitung dan dibulatkan ke ribuan terdekat
        total_price = round((p_base * multiplier) / 1000) * 1000
        
        # Hitung besarnya biaya tambahan (extra labor/material)
        extra_cost = total_price - p_base

        result = {
            "panel": panel_name,
            "base_price": p_base,
            "extra_labor_cost": extra_cost,
            "total_estimated_cost": total_price,
            "formatted_base": f"Rp {p_base:,.0f}".replace(",", "."),
            "formatted_total": f"Rp {total_price:,.0f}".replace(",", "."),
            "parameters": {
                "mask_area_ratio": round(a_mask, 3),
                "depth_deviation": round(d_depth, 3),
                "multiplier": round(multiplier, 3)
            }
        }
        
        return result, "Success"

# --- Pengujian Fungsi Matematis ---
if __name__ == "__main__":
    calculator = CostCalculator(price_list_path='data/price_list.json')
    
    # Simulasi Kasus 1: Pintu Depan penyok kecil tapi sangat dalam
    # Luas mask hanya 5% (0.05), tapi kedalaman penyok tinggi (0.75)
    res, status = calculator.calculate_cost(
        panel_name="pintu_depan",
        mask_area_ratio=0.05,
        depth_deviation=0.75
    )
    
    if res:
        print("=== HASIL KALKULASI FORMULA MATEMATIS ===")
        print(f"Panel              : {res['panel']}")
        print(f"Harga Base Panel   : {res['formatted_base']}")
        print(f"Rasio Luas (A_mask): {res['parameters']['mask_area_ratio']}")
        print(f"Kedalaman (D_depth): {res['parameters']['depth_deviation']}")
        print(f"Pengganda (Mult)   : {res['parameters']['multiplier']}x")
        print(f"TOTAL ESTIMASI BIAYA: {res['formatted_total']}")
    else:
        print(f"Error: {status}")