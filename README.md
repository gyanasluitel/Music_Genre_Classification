# Music_Genre_Classification
A Musical Genre Classifier developed using Deep Learning Models

-----------------------------------------------------------------------------

OS Requirement: Windows (64 Bit)

-----------------------------------------------------------------------------

-------------------------------------------------------------------
Installing Dependencies and running project
------------------------------------------------------------------
Make sure Python and pip is installed (64 bit).

Steps:
Open the command prompt in this project directory.

---------------Installing pipenv and virtualenv-------------
$ pip install pipenv

$ pip install virtualenv

--------------Creating virtualenv and activating------------
$ virtualenv <env_name>
Eg: virtualenv music_test

$ music_test\Scripts\activate

--------------Install packages----------------------
$ pipenv install -r final_requirements.txt


--------------Run Streamlit Application To Predict Songs--------------
$ cd src
$ streamlit run music_genre_app.py