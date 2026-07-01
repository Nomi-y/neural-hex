# Even more UI and also backend

I am pretty happy with the recent changes but there are a few more touch-ups needed to the ui
the networking issue is now solved and i have successfully played against an opponent on the local network

## server name

for better networking support when deployed later add an environment variable for the servers fqdn and and backends fqdn (ill set it to something like hex.icelabs.dev or similar)
The backends fqdn is for the api calls and admin panel this is why its needed
Keep the setting flexible so that i can test with ip addresses and localhost without issue

## invite link

add an option to join with invite link/game code to the section of the UI
the added server name is used when generating invite link
the copy feature is broken it fails 100% of the time

## Board size setting

When the screen is zoomed in or smaller the selection for board size overflows. 
If the entire selection bar cant fit it should turn into an autoscrolling slider that tries to center the selected option

## UI Colors

Add administrative options for changing UI colors on the fly, integrate this where best fit. This feature will mostly be used to protoype UI designs and will probably be removed for any full release
Whether to implement this in the Admin page, as env vars or a config file is up to you, the requirement is rapid prototyping of color schemes

## Board

### Edges

The edges are clearly separated into a lot of individual rectangular segments that don't meet

### Recent move

Not a big fan of the yellow dot showing the most recent move, changing the entire pieces color to an accept based on the player color might be a better idea

### Z- fighting

Clear Z layering is needed to avoid visual glitches based on the fixes implemented here

### Planning mode

The planning hexes should also stay inside the tile gaps and not override them
When left clicking while any plan is drawn do not submit/play a move, it should only erase plans

## Backend

Once all the UI changes are done I want some changes to the backend for easier development

### Engines

Create a subdirectory ./Backend/Source/Engines/Engines/
Any file in this directory exports a default implementing a base engine driver.
This way engine development can be kept organised when I want to add more engines

### GameView

The Game view should have some enriched types for easier engine development, right now Cells is a number[] but without documentation it does nothing for me

### Managed engines

Managed engines are added via code so they can be removed from the code in the Admin page

### External engines

They are a great feature, add some documentation directly in the Admin page that explains how to connect external engines using the generated url

## General issues

I am running into token spend limits so we will collaborate in the future. after you implement the changes in this file development will continue as follows

I will work on the backend as much as possible by myself while you handle the UI code and design.
