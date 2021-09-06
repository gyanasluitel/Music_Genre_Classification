# Music_Genre_Classification
A Musical Genre Classifier developed using Deep Learning Models

-----------------------------------------------------------------------------

OS Requirement: Windows (64 Bit)

-----------------------------------------------------------------------------

Installing Dependencies and running project
------------------------------------------------------------------
Make sure Python and pip is installed (64 bit).

Steps:
1. Open the command prompt in this project directory.

2. Installing pipenv and virtualenv
    $ pip install pipenv

    $ pip install virtualenv

3. Creating virtualenv and activating
    $ virtualenv <env_name>
    Eg: virtualenv music_test

    $ music_test\Scripts\activate

4. Install packages
    $ pipenv install -r final_requirements.txt

5. Run Streamlit Application To Predict Songs
    $ cd src
    $ streamlit run music_genre_app.py
