# Improving the user interface

The goal is to have a responsive User interface for desktop clients.

## Colors

I am not a big fan of the current color scheme. Dark mode is fine but it is bland. If you could implement a new theme based on pastel colors it would already set the site apart.
There is no need for a light mode toggle, the theme can be baked in as a unified single color scheme
Player colors on the other hand could be customized later so these should be kept separate from the rest.

## Responsiveness

Every single action (must have an equal and opposite reaction)... every action should give feedback to the client - sound design is not needed but a fail and a success should be easy to see, and it should be as easy to figure out what failed.

## Board

* Instead of a line showing the players borders, the border can be integrated by coloring the edges of the outside hexagons
* The last placed hex by any player shows with a weird white outline, displaying which move was last is alright but maybe with a different indicator

## Design 

* Board size should not be a slider, instead a a horizontal row of radio buttons - and disappear entirely if there is only one option to choose from
* Options are currently checklist style, I want a fun design that sets itself apart from this boring feel
* Other buttons should also show the playfulness, the main goal is to get rid of vertical lists for everything
* For showing contrast you can use the border of objects, pairing with this a border that is a bit thicker than the current one could make for an interesting design
* Remove alerts and replace them with a custom popup

## Match

When playing with time limit, adding time should not increase the timer over the initial value, plaxing 5+5 should never go above 5 minutes remaining
Display players, match controls (Resign) and (add this) move history on the left side of the board instead of splitting them up
If there is not enough horizontal space, then move this module below the board
Interacting with the canvas using the right mouse button could show move planning like on chess websites

## Administration

The admin should be able to configure what kind of matches are available to queue for, this should reflect in the backend and persistence
For this include an export/import to/from JSON option

