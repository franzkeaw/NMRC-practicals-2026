This repository contains the computer practicals for the course
Neural Models, Representation and Consciousness, as part of the 
Master Biomedical Sciences, track Cognitive Neurobiology and Clinical Neurophysiology, 
of the University of Amsterdam.

A special thanks to Pietro Marchesi  ( https://github.com/pietromarchesi ), the original creator and mastermind of these practicals.


# Requirements
The only thing you need to follow these computer practicals is (1) a webbrowser, and (2) a google account: <br>

For the practicals, we will rely on using python, which is a popular programming language that makes it easy 
to write and understand code.
Furthermore, we will use Jupyter Notebooks, which is a special tool that lets you write and run Python code in small,
easy-to-read sections in your web browser. It’s like a digital notebook where you can mix text, 
(python-)code, and results all in one place. We have created one notebook for each practical. <br>

You can "run" these notebooks through your webbrowser in google colab. 
The downside is that you will need a google account to make use of colab. The (large) advantage is that you don't 
need to install python, conda, make python environments etc., which usually is a large source of issues and heachaches.

# Instructions:

## (1) Create google account
If you dont have one yet: Create a google account <br>
[ If you absolutely don't want to create a google-account, you can use the pyproject.toml from the github link to create
a custom env and run things locally on your laptop, if you are familiar with python, environments and jupyter notebooks. ]

## (2) Open google colab
Go to https://colab.research.google.com/ <br>
Usually, a window opens automatically that asks you to open a notebook. <br>
(If not: go to the left top in the brwoser and click "File"  -> "Open notebook") <br>

## (3) Get a notebook from github and open it
In the window, go to "GitHub", and enter the following address: https://github.com/franzkeaw/NMRC-practicals-2026
(searching might take a few seconds). Now you should see a list of notebooks, click on the first one
(called "0_getting_started.ipynb" ). This should open the file in the browser. 

## (4) Copy this notebook to your own google drive
Important!  Before you do anything else, you need to copy this notebook (which you just opened) to your own google-drive. 
Google colab has just opened this notebook from the GitHub repository, so when you want to make changes
(e.g. typing in your answer),you will not be able to save those changes, unless you first copy the notebook to your
own drive: <br>
In top-left menu, click "File"->"Save a copy in drive".  <br>

The next time you want to open this notebook, you can do so directly from your own drive (so you choose "drive"
instead of "Github" after clicking "Open notebook", in step (3))

## (5) Go through the notebook
Now you can work with the Notebooks. How do they work? <br>
They consist of "cells", which can either contain plain text (like the 
descriptions, or your answer) or python code. The cells which contain python code can be "run" (or executed) to 
perform a certain action, e.g. to calculate something or to display some data plots. You can run such a cell by clicking
on the small "play" button in the top left corner of each cell. <br>

For each notebook, it is important that you always run the very first cell in the top, before you start doing anything else.
That is because the first cell tells the notebook which python code to import to make the other cells run. 
Note that this "import" is only alive as long as you have the notebook open. If you close it, or shutdown your laptop and 
open the notebook the next day, you need to run the first cell again, before you can run any other code in the notebook. 

Go ahead and open the first notebook "0_getting_started.ipynb", run the first cell and then continue reading
through the rest of the notebook, following the instructions in there.