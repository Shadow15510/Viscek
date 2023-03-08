# ┌──────────────────────────────────┐ #
# │        Vicsek GUI — 1.0.0        │ #
# │ Alexis Peyroutet & Antoine Royer │ #
# │ GNU General Public Licence v3.0+ │ #
# └──────────────────────────────────┘ #

import tkinter as tk
import vicsek


# ┌─────────┐ #
# │ Classes │ #
# └─────────┘ #
class VicsekGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.geometry("300x300")
        self.title("Paramètres du modèle Vicsek")
        self.resizable(False, False)


app = VicsekGUI()
app.mainloop()