"""Sphinx configuration for the edgeofpy documentation."""

from importlib.metadata import version as _version

project = 'edgeofpy'
author = "Jordan O'Byrne"
copyright = f'2023-2026, {author}'
release = _version('edgeofpy')            # e.g. '0.1.0' or '0.2.0.dev0'
version = '.'.join(release.split('.')[:2])

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'numpydoc',
    'myst_parser',                        # to include CHANGELOG.md
]

# API reference: one page per function, generated from the docstrings
autosummary_generate = True
autodoc_default_options = {'members': True}
numpydoc_show_class_members = False
numpydoc_xref_param_type = False
default_role = 'py:obj'                   # `name` links to that object

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
}

source_suffix = {'.rst': 'restructuredtext', '.md': 'markdown'}
exclude_patterns = ['_build']

html_theme = 'pydata_sphinx_theme'
html_title = f'edgeofpy {release}'
html_theme_options = {
    'github_url': 'https://github.com/jnobyrne/edgeofpy',
    'icon_links': [{'name': 'PyPI',
                    'url': 'https://pypi.org/project/edgeofpy/',
                    'icon': 'fa-brands fa-python'}],
    'navigation_with_keys': False,
    'show_toc_level': 2,
}
html_context = {'default_mode': 'auto'}  # follow the reader's light/dark mode
