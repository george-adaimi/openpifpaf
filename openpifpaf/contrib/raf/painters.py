import numpy as np

try:
    import matplotlib
    import matplotlib.animation
    import matplotlib.collections
    import matplotlib.patches
except ImportError:
    matplotlib = None

class RelationPainter:
    def __init__(self, *, xy_scale=1.0):
        self.xy_scale = xy_scale

    def annotations(self, ax, annotations, *,
                    color=None, colors=None, texts=None, subtexts=None):
        for i, ann in reversed(list(enumerate(annotations))):
            this_color_obj = ann.category_id_obj - 1
            this_color_sub = ann.category_id_sub - 1
            if colors is not None:
                this_color_obj = colors[i]
                this_color_sub = colors[i]
            elif color is not None:
                this_color_obj = color
                this_color_sub = color

            text_sub = ann.category_sub
            text_obj = ann.category_obj
            text_rel = ann.category_rel
            if texts is not None:
                text_sub = texts[i]
                text_obj = texts[i]
                text_rel = texts[i]

            subtext_sub = None
            subtext_obj = None
            subtext_rel = None
            if subtexts is not None:
                subtext_sub = subtexts[i]
                subtext_obj = subtexts[i]
                subtext_rel = subtexts[i]
            elif ann.score_obj:
                subtext_obj = '{:.0%}'.format(ann.score_obj)
                subtext_sub = '{:.0%}'.format(ann.score_sub)
                subtext_rel = '{:.0%}'.format(ann.score_rel)

            self.annotation(ax, ann, color=(this_color_sub, this_color_obj), text=(text_sub, text_rel, text_obj), subtext=(subtext_sub, subtext_rel, subtext_obj))

    def annotation(self, ax, ann, *, color=None, text=None, subtext=None):
        if color is None:
            color_sub = 0
            color_obj = 0

        if isinstance(color[0], (int, np.integer)):
            color_sub = matplotlib.cm.get_cmap('tab20')((color[0] % 20 + 0.05) / 20)
        if isinstance(color[1], (int, np.integer)):
            color_obj = matplotlib.cm.get_cmap('tab20')((color[1] % 20 + 0.05) / 20)

        # SUBJECT
        x, y, w, h = ann.bbox_sub * self.xy_scale
        if w < 5.0:
            x -= 2.0
            w += 4.0
        if h < 5.0:
            y -= 2.0
            h += 4.0

        # draw box SUBJECT
        ax.add_patch(
            matplotlib.patches.Rectangle(
                (x, y), w, h, fill=False, color=color_sub, linewidth=1.0))

        # draw text SUBJECT
        ax.annotate(
            text[0],
            (x, y),
            fontsize=8,
            xytext=(5.0, 5.0),
            textcoords='offset points',
            color='white', bbox={'facecolor': color, 'alpha': 0.5, 'linewidth': 0},
        )

        # SUBJECT
        if subtext is not None:
            ax.annotate(
                subtext[0],
                (x, y),
                fontsize=5,
                xytext=(5.0, 18.0 + 3.0),
                textcoords='offset points',
                color='white', bbox={'facecolor': color, 'alpha': 0.5, 'linewidth': 0},
            )

        # OBJECT
        x, y, w, h = ann.bbox_obj * self.xy_scale
        if w < 5.0:
            x -= 2.0
            w += 4.0
        if h < 5.0:
            y -= 2.0
            h += 4.0

        # draw box OBJECT
        ax.add_patch(
            matplotlib.patches.Rectangle(
                (x, y), w, h, fill=False, color=color_obj, linewidth=1.0))

        # draw text OBJECT
        ax.annotate(
            text[2],
            (x, y),
            fontsize=8,
            xytext=(5.0, 5.0),
            textcoords='offset points',
            color='white', bbox={'facecolor': color, 'alpha': 0.5, 'linewidth': 0},
        )

        # OBJECT
        if subtext is not None:
            ax.annotate(
                subtext[2],
                (x, y),
                fontsize=5,
                xytext=(5.0, 18.0 + 3.0),
                textcoords='offset points',
                color='white', bbox={'facecolor': color, 'alpha': 0.5, 'linewidth': 0},
            )
