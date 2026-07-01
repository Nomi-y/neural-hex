# Redoing the frontend for my web application

I am unhappy with the design of the current frontend of my web application and would like to redo it.
Below is a list of features and considerations for this task

## General information

In order to do this task well you are given a suite of tools at your disposal

* Web development plugins are installed and available
* The project root contains ./docker-compose.yml which can be used to spin up a full dev stack
* a detailled summary of past changes to both frontend and backend can be found in ./summary.txt
* For coding conventions refer to ./.claude/Context.sh

## Permissions

You have full permission and access to this project and have the green light for running git commands as well
The current branch is main, from now on I would like to use branches for features and rebase them into main after verifying their functionality
permission to manage project dependencies with the bun package manager is also granted

## Goals

Redo the entire frontend UI
Feel free to change the current layout completely
since i cannot tell you what i should look like i will instead talk about the intent behind this application.
The application is a way for people to play the board game hex online
The general feel of the ui should be similar to websites like chess.com or lichess.org

I want for interactible elements to feel responsive, use the following ideas:

* animations using exponential curves
  * this includes movement and transitions
  * hover effects
* color changing, indicate intent with accent but don't overdo it
* no default/unstyled elements

Remember to use animations with great care.
an overanimated website that does too much can feel overwhelming

## Tech stack

the tech stack for the frontend is up to your discretion
the current stack is bun+react+vite
The online requirement is bun with typescript

## Singe page/multi page appliction

This is up to your discretion looking at the scope and idea behind this project i know that you'll make the correct decision there

## The game board

the game board is one element that I am pretty happy with how it currently looks.
This is the one element I would like to keep similar to how it is

## Icon

./hexagon.png is a placeholder icon that can be used as a facicon and logo for the website.
Move it to a location that makes sense in the project

## colors

You will find the color scheme of the existing page to be created in a strage way.
The purpose is to control color schemes with an admin panel (that exists) for rapid color prototyping
Keep this in mind when you are thinking about changing css color variable names or creating new ones

## The admin panel

somewhere in the project is an admin panel. leave this as-is.


