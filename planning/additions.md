# Next steps

Now that base functionality is here next I would like you tto implement the following changes and features

## Changes

### Persistence

I am not quite happy with SQLite as the persistence layer, it was a sensible decision but something tell me that a different DB would do better for this project.
Do some research about what DBs other large games use and remove SQLite from the picture
If Dependency injection for the DB and ORM is not implemented yet, look into some lightweight options for that too - no raw SQL or other data languages

### More data

Storing a move history is a smart decision.
Implement the move history in a way where you can easily get all relevant data for analysing a match later

### swap

swapping a core part of what drew me to hex so adding swapping is a no-brainer

### Options for match creation

Swapping being an option and picking what color you play as should be changeable in match creation. Random color is also a valid choice, like in modern chess games

### playerid

My thought behind this is if i add accounts later I'll need a way to identify players. Don't add login features or other account features yet, i just want the system to be in place

## Features

### engines

This is up to your discretion - I will add engines later.
I want to create a way for not only the administrator to add engines but also for players to register engines - even self-hosted engines.
If you need an engine stub to test features just create the simplest dumbest engine that always plays on the next free tile in the sequence
IMPORTANT: If you evaluate a feature of this scale to be unfeasible with the current architecture let me know and DO NOT implement this feature.
An idea on my part would be to use inheritance (gasp) and derive player and engine from a common entity where the limit per match is 1 for a player and infinite for an engine

Thoughts for later/Note to self:  
If this works and engines can be added it should be pretty easy to add an egine orchestrator to the server that can manage instances of engines running and match players to engines.

### clock

If rehydration is a serious issue that could hit me on a regular basis or on user error then it needs to be looked at. I don't plan to scale huge however and if it allows me to swap out servers for an update ill take it
