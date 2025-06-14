"""
Setup script for Autotrader Bot
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith('#') and not line.startswith('talib')
        ]

setup(
    name="autotrader-bot",
    version="0.1.0",
    author="Autotrader Development Team",
    author_email="autotrader@example.com",
    description="Continuous Learning Cryptocurrency Trading Bot",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/autotrader-bot",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "pre-commit>=2.20.0",
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
            "ipywidgets>=8.0.0",
        ],
        "talib": [
            "talib-binary>=0.4.19",
        ]
    },
    entry_points={
        "console_scripts": [
            "autotrader=autotrader.cli:main",
            "autotrader-bot=autotrader.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "autotrader": [
            "config/*.yaml",
            "config/*.yml", 
            "config/*.ini",
        ],
    },
    zip_safe=False,
    keywords="cryptocurrency trading bot machine learning tensorflow btc bitcoin",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/autotrader-bot/issues",
        "Source": "https://github.com/yourusername/autotrader-bot",
        "Documentation": "https://autotrader-bot.readthedocs.io/",
    },
)
