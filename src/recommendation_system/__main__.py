import os
import sys

from streamlit.web import cli as stcli

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'app.py')

sys.argv = ["streamlit", "run", filename]
sys.exit(stcli.main())
