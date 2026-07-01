# Task list 9

Summary:
These tasks mainly focus on backend changes, changing where the data lives
A small UI feature is to be implemented as well
In addition a have discovered a graphical issue that needs fixing

After your changes are comitted and verified do not take the stack down, i dont want to spin it up after it just got taken down

## Where data lives

### The match list

The frontend is displaying some matches to queue for
The admin page has a JSON field that allows customizing the matches
If this data is not persisted in the DB then persist it and make that the single source of truth
Result: When displaying the list of different matches to queue for fetch the data from the db or a cache, there are to be no hardcoded matches

## Graphical issue

Clock selection has a small visual bug
The horizontal radio pills (term coined by me) has the following styling issue:
When switching the format the body of the parent element containing the button expands to the width of the explanatory text making the size inconsistent for each selection

## UI Features

Some of these add new data metrics - they will need http endpoints

### total of players online and total live matches

Add these metrics to the nav bar:

* How many players are online right now
* How many matches are being played right now

The number should update every 5-10 seconds
For best control add a setting to the admin panel to control update speed
Persist settings from the admin panel in the database (if needed)

### Board size

The servers allowed board size should be changeable in the admin panel
Add fields for min/max + a save button that persists changes

## The admin panel

I am adding this section just to sum up desired behaviour
I am sure some of the persistence already exists but its better to check than to guess

### More settings

Add meaningful settings to the admin panel where it makes sense.
To decide: Does it make sense to change this setting live?
If yes it goes in the admin panel

### Persistence

all server settings in the Admin panel except the color changer should survive a restart

### UI

Split the Admin panel into meaningful separate tabs with a navbar
No additional styling is required for the admin panel it exists to adjust server settings not to look good to users
Once the panel is split up add some search filters to users and engines
