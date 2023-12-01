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

sys.path.insert(0, os.path.abspath("../"))

# -- Project information -----------------------------------------------------

project = "SCOPE-RL"
copyright = "2023, Haruka Kiyohara, Ren Kishimoto, HAKUHODO Technologies Inc., Hanjuku-kaso Co., Ltd"
author = "Haruka Kiyohara"

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
    "sphinx.ext.githubpages",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "sphinx_favicon",
    # "sphinx_gallery.gen_gallery",
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
html_logo = "_static/images/logo.png"
html_context = {
    "default_mode": "light",
}
html_theme_options = {
    "github_url": "https://github.com/hakuhodo-technologies/scope-rl",
    # "twitter_url": "https://twitter.com/{account}",
    "icon_links": [
        {
            "name": "Speaker Deck",
            "url": "https://speakerdeck.com/harukakiyohara_/scope-rl",
            "icon": "fa-brands fa-speaker-deck",
            "type": "fontawesome",
        },
    ],
    "header_links_before_dropdown": 4,
    "navbar_start": ["navbar-logo"],
    # "navbar_start": ["navbar-logo", "version"],  # causes some errors
    "footer_items": ["copyright"],
    "show_prev_next": False,
    # "google_analytics_id": "UA-XXXXXXX",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

html_title = "SCOPE-RL"
html_use_opensearch = "https://scope-rl.readthedocs.io/en/latest/"
favicons = [{"href": "images/favicon.png"}]


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
    "scope_rl.policy.head.BaseHead": False,
    "scope_rl.policy.head.ContinuousEvalHead": False,
    "scope_rl.policy.head.GaussianHead": False,
    "scope_rl.policy.head.TruncatedGaussianHead": False,
    "scope_rl.policy.head.EpsilonGreedyHead": False,
    "scope_rl.policy.head.SoftmaxHead": False,
    "scope_rl.policy.head.OnlineHead": False,
    "scope_rl.ope.weight_value_learning.function.VFunction": False,
    "scope_rl.ope.weight_value_learning.function.StateWeightFunction": False,
    "scope_rl.ope.weight_value_learning.function.DiscreteQFunction": False,
    "scope_rl.ope.weight_value_learning.function.ContinuousQFunction": False,
    "scope_rl.ope.weight_value_learning.function.DiscreteStateActionWeightFunction": False,
    "scope_rl.ope.weight_value_learning.function.ContinuousStateActionWeightFunction": False,
}
# numpydoc_xref_param_type = True
# numpydoc_xref_aliases = {
#     "OffPolicyEvaluation": "scope\_rl.ope.ope.OffPolicyEvaluation",
#     "CumulativeDistributionOPE": "scope\_rl.ope.ope.CumulativeDistributionOPE",
#     "OffPolicySelection": "scope\_rl.ope.ops.OffPolicySelection",
#     "BaseOffPolicyEstimator": "scope\_rl.ope.estimators_base.BaseOffPolicyEstimator",
#     "BaseCumulativeDistributionOPEEstimator": "scope\_rl.ope.estimators\_base.BaseCumulativeDistributionOPEEstimator",
#     "SyntheticDataset": "scope\_rl.dataset.synthetic.SyntheticDataset",
#     "TrainCandidatePolicies": "scope\_rl.policy.opl.TrainCandidatePolicies",
# }

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "d3rlpy": ("https://d3rlpy.readthedocs.io/en/v1.1.1/", None),
    "gym": ("https://www.gymlibrary.dev/", None),
    "gymnasium": ("https://gymnasium.farama.org/", None),
}

# # gallery example path
# from sphinx_gallery.sorting import ExplicitOrder
# from sphinx_gallery.sorting import FileNameSortKey

# sphinx_gallery_conf = {
#     "examples_dirs": "example",  # path to your example scripts
#     "gallery_dirs": "documentation/_autogallery",  # path to where to save gallery generated output
#     "subsection_order": ExplicitOrder(
#         [
#             "example/basic_ope",
#             "example/cumulative_distribution_ope",
#             "example/ops",
#             "example/scope_rl_others",
#             "example/multiple_datasets",
#             "example/rtbgym",
#             "example/footer",
#         ]
#     ),
#     "within_subsection_order": FileNameSortKey,
#     "download_all_examples": False,
# }
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
