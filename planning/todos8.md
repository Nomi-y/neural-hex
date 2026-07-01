# Changes to logic and queue

This change focuses on two key parts of the web app as well as fixing some bugs
For frontend work use the plugins at your disposal
Additional information is in ./coding-conventions.md and more info can be gained by running ./.claude/Context.sh

## The clock

The game clock currently has a strict min+sec format
The following changes are to be implemented:

### Internal values

All internal values of the clock are in ms.
This allows the admin user to set more detailled time constraints at the cost of having to convert minutes to ms

### Clock formats

Currently the clock supports Fischer
Add support for a couple popular clock options

### User input

When creating a timed friend match a user should be able to do the following interactions:

* Select the added Clock modes and either have a visual representation or some text explaining how the clock behaves in selected format
* Clicking the time suffix (min/sec) toggles between these options
* ability to type numbers into the displays number fields

## Match Queue

A user should be able to queue for multiple different formats/matches at once.
Example behaviour with common lobby names for board games

1. Click on the button to queue for a Blitz match
2. Square updates and visually indicates that the player is queued for Blitz
3. Queue for Rapid and Bullet just like Blitz
4. Click on Blitz again cancelling waiting for these matches.
5. A match is found, all other waits are cancelled and the player is moved into a game

For convenience, once more than 2 queues are selected at once show a button for cancelling all searches
This instantly cancels all queues without confirmation
Make sure the button appearing does not look jarring

## Visual Bug

I have discovered a visual glitch that occurs sometimes when logging out.
Clicking the log out button does not update the page
Troubleshoot this issue, and make sure to avoid similar bugs implementing the new queue
