"""
ver_letras.py — Visor de las señas dinámicas guardadas (j, k, q, x, z, ñ)
Genera un GIF animado por letra con el esqueleto de la mano (orden correcto + aspect equal).
Corre: python ver_letras.py
"""
import os, glob, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation

DYN = os.path.join('data_nuevos', 'dynamic_letters')
OUT = 'gifs_letras'
os.makedirs(OUT, exist_ok=True)

CONN = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),
        (10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]

def animar(letra):
    files = sorted(glob.glob(os.path.join(DYN, letra, '*.npy')))
    if not files:
        print(f'[!] sin datos: {letra}'); return None
    seq = np.load(files[0])  # (30, 126)

    fig, ax = plt.subplots(figsize=(4,4.5))
    def draw(fr):
        ax.clear()
        ax.set_title(f'Letra {letra.upper()}  ({fr+1}/30)', fontsize=15, weight='bold')
        ax.set_xlim(-0.8, 0.8); ax.set_ylim(0.3, -1.2)
        ax.set_aspect('equal'); ax.axis('off')
        frame = seq[fr]
        left  = frame[0:63].reshape(21,3)
        right = frame[63:126].reshape(21,3)
        for hand, color in [(left,'tab:blue'), (right,'tab:red')]:
            if np.all(hand[1:] == 0):
                continue
            p = hand[:, :2]
            for a,b in CONN:
                ax.plot([p[a,0],p[b,0]], [p[a,1],p[b,1]], color=color, lw=2.5)
            ax.scatter(p[1:,0], p[1:,1], c=color, s=30, zorder=3)
            ax.scatter([0],[0], c='black', s=80, zorder=4)  # muñeca
        return []

    anim = animation.FuncAnimation(fig, draw, frames=30, interval=80)
    out = os.path.join(OUT, f'{letra}.gif')
    anim.save(out, writer='pillow', fps=12)
    plt.close()
    return out

if __name__ == '__main__':
    for letra in ['j', 'k', 'q', 'x', 'z', 'ñ']:
        g = animar(letra)
        if g:
            print(f'OK: {g}')
    print(f'\nGIFs en: {os.path.abspath(OUT)}')
