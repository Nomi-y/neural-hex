# Next steps now that requested features work

Before expanding the app, iron out some kinks (not necessarily bugs) first

## Random matches

Random matches currently offer the full selection of options for a game.
This will very likely immediately stall waiting for many players.
Proposed change:
Allow only a certain amount of preset options (3 for now) when choosing a random match.

## Ranked play

Additionally if not implemented yet, add extensibility to the match maker so that later on I can change random match to be Ranked match (It will stay as random match for now).
If a feature doing this already exists, ignore the directive.

## Engine play

Playing against an engine should be with unlimited time.
When changing the board size, make sure the engine selected supports differently sized boards, extend the engine struct as needed

## Unlimited matches

Unlimited time is nice but clenup is nicer.
Matches where one player is disconnected or idle for over X amount of time are forfeit.
To implement this, for now keep it simple but keep an addition of an admin dashboard in mind where it could be changed later.
For testing and ease of testing an environment variable for this will do
