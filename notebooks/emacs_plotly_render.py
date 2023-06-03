"""A utils file from Adria"""

import hashlib
import os

import plotly.io as pio


class EmacsRenderer(pio.base_renderers.ColabRenderer):
    save_dir = "ob-jupyter"
    base_url = f"http://localhost:8888/files"

    def to_mimebundle(self, fig_dict):
        html = super().to_mimebundle(fig_dict)["text/html"]

        mhash = hashlib.md5(html.encode("utf-8")).hexdigest()
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        fhtml = os.path.join(self.save_dir, mhash + ".html")
        with open(fhtml, "w") as f:
            f.write(html)

        return {"text/html": f'<a href="{self.base_url}/{fhtml}">Click to open {fhtml}</a>'}


pio.renderers["emacs"] = EmacsRenderer()


def set_plotly_renderer(renderer="emacs"):
    pio.renderers.default = renderer
