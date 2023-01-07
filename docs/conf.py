# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------

project = "OFRL"
copyright = "2023, Hakuhodo Technologies"
# copyright = "2023, Haruka Kiyohara, Yuta Saito, and negocia, Inc"
author = "Haruka Kiyohara, Kosuke Kawakami, Yuta Saito"

# The full version, including alpha/beta/rc tags
version = "latest"
release = "latest"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autodoc.typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
    "numpydoc",
    "sphinx_design",
    "sphinx_tabs.tabs",
    "sphinxemoji.sphinxemoji",
]

# bibtex
bibtex_bibfiles = ["refs.bib"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/negocia-inc/ofrl",
    # "twitter_url": "https://twitter.com/{account}",
    "icon_links": [
        {
            "name": "Speaker Deck",
            "url": "https://speakerdeck.com/aiueola/ofrl-designing-an-offline-reinforcement-learning-and-policy-evaluation-platform-from-practical-perspectives",
            "icon": "fa-brands fa-speaker-deck",
            "type": "fontawesome",
        },
    ],
    "header_links_before_dropdown": 6,
    # "navbar_start": ["navbar-logo", "version"],
    "footer_items": ["copyright"],
    "show_prev_next": False,
    # "google_analytics_id": "UA-XXXXXXX",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

html_title = "OFRL"

# whether to display to the source .rst file
html_show_sourcelink = False
html_show_sphinx = False

autosummary_generate = True
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_default_options = {
    "members": None,
    "member-order": "bysource",
    "exclude-members": "forward,close,render,np_random,render_mode,unwrapped,spec",
}

# mapping between class methods and its abbreviation
numpydoc_member_order = "bysource"
numpydoc_show_inherited_class_members = {
    "ofrl.policy.head.BaseHead": False,
    "ofrl.policy.head.ContinuousEvalHead": False,
    "ofrl.policy.head.ContinuousGaussianHead": False,
    "ofrl.policy.head.ContinuousTruncatedGaussianHead": False,
    "ofrl.policy.head.DiscreteEpsilonGreedyHead": False,
    "ofrl.policy.head.DiscreteSoftmaxHead": False,
    "ofrl.policy.head.OnlineHead": False,
    "ofrl.ope.weight_value_learning.function.VFunction": False,
    "ofrl.ope.weight_value_learning.function.StateWeightFunction": False,
    "ofrl.ope.weight_value_learning.function.DiscreteQFunction": False,
    "ofrl.ope.weight_value_learning.function.ContinuousQFunction": False,
    "ofrl.ope.weight_value_learning.function.DiscreteStateActionWeightFunction": False,
    "ofrl.ope.weight_value_learning.function.ContinuousStateActionWeightFunction": False,
}
numpydoc_xref_aliases = {
    # 'LeaveOneOut': 'sklearn.model_selection.LeaveOneOut',
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "d3rlpy": ("https://d3rlpy.readthedocs.io/en/v1.1.1/", None),
    "gym": ("https://www.gymlibrary.dev/", None),
    "gymnasium": ("https://gymnasium.farama.org/", None),
}

# gallery example path
from sphinx_gallery.sorting import ExplicitOrder
from sphinx_gallery.sorting import FileNameSortKey

sphinx_gallery_conf = {
    "examples_dirs": "tutorial",  # path to your example scripts
    "gallery_dirs": "documentation/_autogallery",  # path to where to save gallery generated output
    "subsection_order": ExplicitOrder(
        [
            "tutorial/basic_ope",
            "tutorial/cumulative_distribution_ope",
            "tutorial/ops",
            "tutorial/ofrl_others",
            "tutorial/rtbgym",
            "tutorial/footer",
        ]
    ),
    "within_subsection_order": FileNameSortKey,
    "download_all_examples": False,
}
# gallery thumbnail
# nbsphinx_thumbnails = {
#     'gallery/thumbnail-from-conf-py': 'gallery/a-local-file.png',
# }

# # provide links for linkcode
# def linkcode_resolve(domain, info):
#     if domain != 'py':
#         return None
#     if not info['module']:
#         return None
#     filename = info['module'].replace('.', '/')
#     return f"https://github.com/{repository}/tree/main/{filename}.py"
