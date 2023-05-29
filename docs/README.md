SCOPE-RL docstring
========

### Prerequisite

Our documentation relies on [Sphinx](https://github.com/sphinx-doc/sphinx). To use, download the following packages through `pip` in advance.

```bash
pip install Sphinx pydata-sphinx-theme numpydoc sphinx_design sphinx-gallery sphinx-tabs sphinx-copybutton sphinxemoji sphinxcontrib-bibtex
```

Please also refer to `./conf.py` for the list of extensions.

```python
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
```

## Preview
To build the docstring in the local environment, run the following command.
```bash
sphinx-build -b html docs docs/preview
```

The top-page is accessible in `./preview/index.html`.

## Useful links
* [Sphinx documentation](https://www.sphinx-doc.org/en/master/)
* [Sphinx PyData Theme documentation](https://pydata-sphinx-theme.readthedocs.io/en/stable/index.html)
* [Sphinx Numpydoc documentation](https://numpydoc.readthedocs.io/en/latest/)
* [Docutils' documentation about ReStructuredText](https://docutils.sourceforge.io/docs/user/rst/quickstart.html)
* [Sphinx Design documentation](https://sphinx-design.readthedocs.io/en/latest/)
* [Sphinx Gallery documentation](https://sphinx-gallery.github.io/stable/index.html)
* [Sphinx Tabs documentation](https://sphinx-tabs.readthedocs.io/en/latest/)
* [Sphinx extensions documentation](https://sphinx-extensions.readthedocs.io/en/latest/index.html)
