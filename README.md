# 🎬 Movie Recommendation System 🍿

[![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Powered by: Streamlit](https://img.shields.io/badge/Powered%20by-Streamlit-red.svg)](https://streamlit.io/)

A simple Streamlit app that recommends movies based on your initial selections! Uses content-based filtering.

Dataset extracted from [Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

![Screenshot](docs/screenshot.png)

## ✨ Features

*   Select up to 5 movies you like via search.
*   Get personalized recommendations based on movie content (genres, runtime, language).
*   Simple and clean Streamlit interface.
*   Uses `uv` for fast dependency management.
        
## 🚀 Getting Started

**1. Clone the Repository** 💾

```bash
git clone https://github.com/PedroG022/recommendation-system
cd recommendation-system
```

**2. Set up Virtual Environment & Install Dependencies (using uv)** 💨

*Make sure `uv` is installed (`pip install uv` or `pipx install uv`).*

```bash
uv sync
```

**3. Activate Virtual Environment** ✅

Choose the command for your shell:

*   **Bash/Zsh:** `source .venv/bin/activate`
*   **Fish:** `source .venv/bin/activate.fish`
*   **Cmd (Windows):** `.venv\Scripts\activate.bat`
*   **PowerShell (Windows):** `.venv\Scripts\Activate.ps1`

**4. Run the App!** ▶️

```bash
recommendation-system
```

---

Enjoy finding your next favorite movie! 🎉
