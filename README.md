Machine Learning Presentation
=============================

Usage
-----

To run the jupyter notebook:

    conda create -n ml-presentation -f environment_base.yml
    source activate ml-presentation
    jupyter notebook

Then visit localhost:8888 in your browser to see the notebook

To run the slideshow:

    jupyter nbconvert 'How to Spot a Bear.ipynb' --to slides --template slides_reveal.tpl --post serve

Then visit browser to see the slideshow
