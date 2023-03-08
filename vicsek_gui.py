# ┌──────────────────────────────────┐ #
# │        Vicsek GUI — 1.0.0        │ #
# │ Alexis Peyroutet & Antoine Royer │ #
# │ GNU General Public Licence v3.0+ │ #
# └──────────────────────────────────┘ #

import tkinter as tk
from tkinter.ttk import Progressbar
from tkinter.messagebox import showinfo, showerror
import vicsek
import math


# ┌─────────┐ #
# │ Classes │ #
# └─────────┘ #
class VicsekGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.geometry("640x360")
        self.title("Paramètres du modèle Vicsek")
        self.resizable(False, False)

        self.position_min = tk.StringVar(value=-25)
        self.position_max = tk.StringVar(value=25)
        self.speed_min = tk.StringVar(value=-2)
        self.speed_max = tk.StringVar(value=2)

        self.sight_min = tk.StringVar(value=5)
        self.sight_max = tk.StringVar(value=10)
        self.field_sight_min = tk.StringVar(value=0)
        self.field_sight_max = tk.StringVar(value=360)

        self.noise_min = tk.StringVar(value=0)
        self.noise_max = tk.StringVar(value=1)
        self.fear_min = tk.StringVar(value=0)
        self.fear_max = tk.StringVar(value=1)

        self.nb_agents = tk.StringVar(value=50)
        self.length = tk.StringVar(value=50)
        self.dim = tk.StringVar(value=2)
        self.nb_frames = tk.StringVar(value=20)

        self.columnconfigure(0, weight=1)
        self.right_pane = tk.Frame(self)
        self.right_pane.columnconfigure(0, weight=1)
        self.right_pane.grid(column=1, row=0, sticky="nw")
        self.left_pane = tk.Frame(self)
        self.left_pane.columnconfigure(0, weight=1)
        self.left_pane.grid(column=0, row=0, sticky="ne")

        self.agent_frame = tk.LabelFrame(self.left_pane, text="Paramètres des agents"); self.agent_frame.grid(column=0, row=0, padx=2, pady=2, sticky="ne")
        button = tk.Button(self.left_pane, text="Valider", relief="raised", command=self.validation)
        button.grid(column=0, row=1, sticky="ew")

        self.settings_frame = tk.LabelFrame(self.right_pane, text="Réglages des vitesses et positions"); self.settings_frame.grid(column=1, row=1, padx=2, pady=2, sticky="nw")
        self.group_frame = tk.LabelFrame(self.right_pane, text="Paramètres du groupe"); self.group_frame.grid(column=1, row=0, padx=2, pady=2, sticky="wn")

        self.create_grp_params()
        self.create_agt_params()
        self.create_scales()

    def create_grp_params(self):
        frame = tk.Frame(self.group_frame)
        frame.rowconfigure(0)
        frame.pack()

        dim = tk.Radiobutton(frame, text="2", value=2, variable=self.dim)
        dim.grid(column=1, row=0)
        dim = tk.Radiobutton(frame, text="3", value=3, variable=self.dim)
        dim.grid(column=2, row=0)

        length = tk.Scale(frame, label="taille de l'espace", from_=1, to=100, length=150, orient=tk.HORIZONTAL, variable=self.length)
        length.grid(column=0, row=0)

        agents = tk.Scale(self.group_frame, label="nombre d'agents", from_=10, to=500, length=300, orient=tk.HORIZONTAL, variable=self.nb_agents)
        agents.pack()

        nb_frames = tk.Scale(self.group_frame, label="nombres d'images", from_=5, to=500, resolution=5, length=300, orient=tk.HORIZONTAL, variable=self.nb_frames)
        nb_frames.pack()

    def create_agt_params(self): 
        frame = tk.Frame(self.agent_frame)
        frame.rowconfigure(0)
        frame.pack()

        sight = tk.Scale(frame, label="distance minimale", from_=0, to=10, length=150, orient=tk.HORIZONTAL, variable=self.sight_min)
        sight.grid(column=0, row=0)

        sight = tk.Scale(frame, label="distance maximale", from_=0, to=10, length=150, orient=tk.HORIZONTAL, variable=self.sight_max)
        sight.grid(column=1, row=0)

        field_sight = tk.Scale(frame, label="angle minimal", from_=0, to=360, length=150, orient=tk.HORIZONTAL, variable=self.field_sight_min)
        field_sight.grid(column=0, row=1)

        field_sight = tk.Scale(frame, label="angle maximal", from_=0, to=360, length=150, orient=tk.HORIZONTAL, variable=self.field_sight_max)
        field_sight.grid(column=1, row=1)

        noise = tk.Scale(frame, label="bruit minimal", from_=0, to=1, digits=3, resolution=0.05, length=150, orient=tk.HORIZONTAL, variable=self.noise_min)
        noise.grid(column=0, row=2)

        noise = tk.Scale(frame, label="bruit maximal", from_=0, to=1, digits=3, resolution=0.05, length=150, orient=tk.HORIZONTAL, variable=self.noise_max)
        noise.grid(column=1, row=2)

        fear = tk.Scale(frame, label="peur minimale", from_=0, to=1, digits=3, resolution=0.05, length=150, orient=tk.HORIZONTAL, variable=self.fear_min)
        fear.grid(column=0, row=3)

        fear = tk.Scale(frame, label="peur maximale", from_=0, to=1, digits=3, resolution=0.05, length=150, orient=tk.HORIZONTAL, variable=self.fear_max)
        fear.grid(column=1, row=3)

    def create_scales(self):
        frame = tk.Frame(self.settings_frame)
        frame.rowconfigure(0)
        frame.pack()

        pos_min = tk.Scale(frame, label="position minimale", from_=-50, to=50, length=150, orient=tk.HORIZONTAL, variable=self.position_min)
        pos_min.grid(column=0, row=0)
        pos_max = tk.Scale(frame, label="position maximale", from_=-50, to=50, length=150, orient=tk.HORIZONTAL, variable=self.position_max)
        pos_max.grid(column=1, row=0)

        spd_min = tk.Scale(frame, label="vitesse minimale", from_=-5, to=5, digits=2, resolution=0.1, length=150, orient=tk.HORIZONTAL, variable=self.speed_min)
        spd_min.grid(column=0, row=1)
        spd_max = tk.Scale(frame, label="vitesse maximale", from_=-5, to=5, digits=2, resolution=0.1, length=150, orient=tk.HORIZONTAL, variable=self.speed_max)
        spd_max.grid(column=1, row=1)

    def validation(self):
        dim, length = int(self.dim.get()), int(self.length.get())
        nb_frames = int(self.nb_frames.get())
        nb_agents = int(self.nb_agents.get())

        sight_min, sight_max = float(self.sight_min.get()), float(self.sight_max.get())
        field_sight_min, field_sight_max = float(self.field_sight_min.get()), float(self.field_sight_max.get())
        noise_min, noise_max = float(self.noise_min.get()), float(self.noise_max.get())
        fear_min, fear_max = float(self.fear_min.get()), float(self.fear_max.get())

        pos_min, pos_max = float(self.position_min.get()), float(self.position_max.get())
        spd_min, spd_max = float(self.speed_min.get()), float(self.speed_max.get())

        field_sight_min = (field_sight_min * math.pi) / 180
        field_sight_max = (field_sight_max * math.pi) / 180

        if (sight_min > sight_max) or (field_sight_min > field_sight_max) or (noise_min > noise_max) or (fear_min > fear_max) or (pos_min > pos_max) or (spd_min > spd_max):
            showerror("Erreur", "Les bornes ne correspondent pas.")

        self.progress_bar = Progressbar(self.left_pane, orient="horizontal", mode="determinate", length=300)
        self.progress_bar.grid(column=0, row=2)

        grp = vicsek.group_generator(
            nb=nb_agents,
            position=(pos_min, pos_max),
            speed=(spd_min, spd_max),
            noise=(noise_min, noise_max),
            sight=(sight_min, sight_max),
            field_sight=(field_sight_min, field_sight_max),
            fear=(fear_min, fear_max),
            length=length,
            dim=dim
        )

        grp.compute_animation(nb_frames, gui=self)
        showinfo("Succès", "Animation exportée au format GIF avec succès.")



app = VicsekGUI()
app.mainloop()