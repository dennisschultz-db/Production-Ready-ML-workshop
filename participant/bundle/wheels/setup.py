"""
setup.py — Build the churn_model wheel for distribution in the workshop.

Build the wheel:
    pip wheel src/churn_model/ --no-deps -w dist/

Upload the resulting .whl to the UC Volume:
    databricks fs cp dist/churn_model-0.1.0-py3-none-any.whl \
        /Volumes/workshop/00_shared/wheels/

Participants install it at the top of their notebooks:
    %pip install /Volumes/workshop/00_shared/wheels/churn_model-0.1.0-py3-none-any.whl
"""

from setuptools import setup, find_packages

setup(
    name="churn_model",
    version="0.1.0",
    description="Telco churn prediction model — Production-Ready ML Systems Workshop",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "scikit-learn>=1.3",
        "mlflow>=2.10",
        "pandas>=1.5",
        "numpy>=1.24",
        "pyyaml>=6.0",
    ],
    entry_points={
        "console_scripts": [
            # Allows running as: python -m churn_model.train (future use)
        ],
    },
)
