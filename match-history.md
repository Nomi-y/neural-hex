# Detailled match history

This directive contains a description for viewing detailled match history
For context read ./coding-conventions.md and use ./.claude/Context.sh

## Match history

The current app contains match history overview
The feature that is to be implemented can be described in the following way:

* Hovering over a match entry highlights it
* Clicking anywhere in the highlighted cell (except the opponent user name) will put the user into a turn by turn replay of the match

## Controls of match viewer

The viewer of the match can be created with many UI elements present but need a bit more control.
Here are the desired controls

* By default the match will begin in paused mode
* Buttons allow the user to play/pause and step
* Play will start automatic match playback at a rate of 1 move per second, customizable in the admin panel and persisted in the database
* Once the match either finishes playing or pause/step is pressed the action will be taken and playback is paused
* Clicking any move in the timeline jumps to that move and pauses playback
* Refreshing the page still shows the analysis and does not reset UI 
  * After an inactivity timeout configurable in the admin page refreshing and revisiting the page will reset the UI
  * To not trap the user in that window include an 'Exit' button 
  * Clicking user profile or play or other elements with transition stops replay

## Server side/client side

By default the client should have all information available to have this match viewer be purlely client side without any server interaction at all (after fetching match data)

The game board should be reused, planning tools should also be available, however gameplay effects like hovering or interacting with the board obviously not

## Engine analysis

One consideration i would like you to take into account is engine analysis.

* How do engines 'analyze' if a play is good or bad?
* I have support for engines that can play the game, if these are not sufficient I will write a directive for implementing analysis engines
* Engine analysis is out of scope for this directive but leaving room to implement support for this in the future would be great

--> To sum these points up i am imagining engine analysis to go like this

1. User requests an analysis of the match
2. The server sends the match to the analysis engine
3. The engine responds and the server gives the analysis to the user to display it (server also stores analysis in the DB to reduce compute)

Depending on the difficulty of implementing analysis features are, write a feature list, summary and thoughts to ./engine-analysis.md
Author it as Claude and make it clear that the text is an analysis that has been performed
