## Steps to get going with this

### Install virtual env and create a profile for kaggle

    pip install https://github.com/pypa/virtualenv/tarball/develop
    virtualenv kaggle
    . kaggle/bin/activate

### Install all the things

    pip install numpy scikit-learn pandas scipy sklearn-pandas

### Generate some entries for the Kaggle Titanic problem

    # Simple version using tutorial from the website
    python titanic.ml
    
    # Version using the ExtraTreesClassifier from scikit-learn
    python titanic-ml.ml
    
Exporation of which classifier would work best can be seen at:

    python titanic-ml-explore.py
