# Engine versus Engine play

I want to add a feature to my hex game that allows a user to pit two engines against each other - or even have an engine play itself

## IMPORTANT

Before you start implementing or even planning get a good view of the project

Important Files:

* README.md
* CLAUDE.md
* coding-conventions.md
* project-rules.md
* summary.txt

These documents contain a clear picture of the current architecture of the project, guidelines in how additions should be architected and a detailled log of all additions and changes made.

## Key features

The backend needs to be able to handle a match with viewers.
Therefore the first imortant feature to be implemented is a **Live Match Viewer** 
Here are some key points / required functionality from the perspective of the end user:

* Joining a match with a link that has already started will allow the user to view the match live
* Expand the nav (Currently just the play tab) view a *View Matches* nav.
* in the live matches nav the user will see:
  1) A random live match (just the board and player names) with a button that when clicked will put the user in the match as a viewer
  2) A button that allows the user to create an engine v engine match

clicking the engine v engine button will bring up a modal with UI similar to the Engine play UI
The user will be allowed to select only options available by both engines.
Timing is always unlimited.
The user is allowed to choose which color starts (or random)

## Admin

Expand the admin page with a radio selection - to enable/disable/logged in users only the feature engine v engine. If the option is disabled the button for engine v engine must not be shown to the user at all

Expand the admin page with additional settings if needed or potentially useful.
Make sure to not add settings just for the sake of settings
