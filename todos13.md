# UI Touchups

## Replay viewer

The replay viewer looks pretty good already but needs a bit more touch ups

### Buttons

Controls work, but the text is unneeded
As for style, combine the control buttons together similar to how other selection style segments work in the ui
No selection animation like the board size 

The button symbols are differently sized, play and step are different from jump to start/end - change jump symbols

### Playing

Instead of a set delay, use the thinking time if available (fixed delay for unlimited matches)

### Move named start

This is unneeded

## UI Behaviour

Add support for forward/back buttons (Mouse4/5, swiping gestures, browsers back and forward buttons)
These should navigate the application as a user would expect it

**IMPORTANT**
If this is not possible to support reasonably for the current UI code let me know; what is needed and how big of a workload this would be
My suspicions are that a single page appliction without routes has little chance

## Gameplay indicator for swap

It would be awesome if you can brainstorm some ideas for how to indicate the swap move in play
The board has no visual indication that a swap has taken place

## View for screenshot

Add a button for the replay viewer that does the following

* Hide the side dock that contains move history and controls
* Show the board taking up as much space as possible
* A single button that when pressed exits this view
* esc and page controls would also exit
* On each tile placed show the turn it has been placed on
* For swap, again indicate this in a way where a single symbol shows that the board has been swapped (and its absence would indicate the opposite)

Summary: Shows the finished board in a state where the entire match can be described by a screenshot of the board
