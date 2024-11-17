deep learning course
==============================

This course is part of a series of modules for data science.
This course assumes you have done the introduction in Python and something similar to the Data Analyses & Visualisation course https://github.com/raoulg/MADS-DAV


The lessons can be found inside the `notebooks`folder.
The source code for the lessons can be found in the `src`folder.

The book we will be using is Understanding Deep Learning. It is available as pdf here: https://udlbook.github.io/udlbook/ but it is highly recommended to buy the book.


Project Organization
------------

    ├── README.md          <- This file
    ├── .gitignore         <- Stuff not to add to git
    ├── .lefthook.yml      <- Config file for lefthook
    ├── pyproject.toml     <- Human readable file. This specifies the libraries I installed to
    |                         let the code run, and their versions.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── processed      <- The processed datasets
    │   └── raw            <- The original, raw data
    │
    ├── models             <- Trained models
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is xx_name_of_module.ipynb where
    │                         xx is the number of the lesson
    ├── presentations      <- Contains all powerpoint presentations in .pdf format
    ├── references         <- background information
    |  └── codestyle       <- Some code Code style standards
    |  └── leerdoelen      <- Learning goals per lesson, including pages to read and videos to watch
    │
    ├── reports            <- Generated analysis like PDF, LaTeX, etc.
       └── figures         <- Generated graphics and figures to be used in reporting

--------

For this project you will need some dependencies.
The project uses python >=3.10 and <3.12, and all dependencies are defined within the `pyproject.toml` file.

The `.lefthook.yml` file is used by [lefthook](https://github.com/evilmartians/lefthook), and lints & cleans the code before I commit it. Because as a student you probably dont commit things, you can ignore it.

I have separated the management of datasets and the trainingloop code. You will find them as dependencies in the project:
- https://github.com/raoulg/mads_datasets
- https://github.com/raoulg/mltrainer

Both of these will be used a lot in the notebooks; by separating them it is easier for students to use the code in your own repositories.
In addition to that, you can consider the packages as "extra material"; the way the pacakges are set up is something you can study if you are already more experienced in programming.

In addition to that, there is a [codestyle repo](https://github.com/raoulg/codestyle) that covers most of the codestyle guidelines for the course.

# Installation
The installation guide assumes a UNIX system (os x or linux).
If you have the option to use a VM, see the references folder for lab setups (both for azure and surf).
For the people that are stuck on a windows machine, please use [git bash](https://gitforwindows.org/) whenever I 
refer to a terminal or cli (command line interface).

## install python with rye
please note that rye might already be installed on your machine.
1. watch the [introduction video about rye](https://rye.astral.sh/guide/)
2. You skipped the video, right? Now go back to 1. and actually watch it. I'll wait.
3. check if rye is already installed with `which rye`. If it returns a location, eg `/Users/user/.rye/shims/rye`, rye is installed.
4. else, install [rye](https://rye.astral.sh/) with `curl -sSf https://rye.astral.sh/get | bash`

run through the rye installer like this:
- platform linux: yes
- preferred package installer: uv
- Run a Python installed and managed by Rye
- which version of python should be used as default: 3.11
- should the installer add Rye to PATH via .profile? : y
- run in the cli: `source "$HOME/.rye/env"`

For windows this should be the same, except for the platform off course...

## add the git repo
run in the cli:

`git clone https://github.com/raoulg/MADS-MachineLearning-course.git`

## add your username and email to git
1. `git config --global user.name "Mona Lisa"`
2. `git config --global user.email "m.lisa@pisa.com"`

## install all dependencies
1. `cd MADS-MachineLearning-course/`
2. `rye sync`

## add your own ssh key
If you want easy access to the VM, add your ssh key to the VM:
1. copy your local ssh key to the VM, see [github docs](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) and/or the azure manual.
2. `cd ~/.ssh`
3. `nano authorized_keys` (or, [if you know how to use it](https://www.youtube.com/watch?v=m8C0Cq9Uv9o), install and use `nvim`)
copy paste your key to the end of the file (leave existing keys) and save the file, then exit
4. check with `cat authorized_keys` that your key is added.

## Still watch the video.

I know some of you still skipped the video. Ok, I get that, but now actually watch it... [introduction video about rye](https://rye.astral.sh/guide/)
