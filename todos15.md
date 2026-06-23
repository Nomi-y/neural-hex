# Ui touchups

This directive is a list of major to minor visual nitpicks That I have noticed. You will be working in parallel to an agent that works on the backend while you fix the frontend.

## Match

### Moves

remove the ↔ icon from the move history swap
before: ↔ swap
after: swap

### Clock

#### Infinite time

The infinity icon is horizontally squished

#### Delay Timer

The +time that is shown when playing with delay is not formatted well, font size and vertical position do not match up

## Replay viewer

### Formatting

When replaying a match i can see the following string

---
nomi won by connection11×11 · 
move 0/46
---

Match information like board size or clock format are redundant and do not need to be displayed

### Screenshot view

The numbers displayed on the hexes when in screenshot view have a similar problem to the Swap icon on the board
Make sure you use a font and styling consistent with the rest of the site

## Board

### Swap icon

icon should use the accent color and have no outline
Make sure it is exactly centered on the stone that was swapped

### Edge geometry

this is one of your weak spots, make sure the edge line indicators/geometry is one continuous line with no self-overlaps of objects

## Match selectcion

### Engine selector

when opening the engine selector after reaching just 4 engines a vertical issue became clear, elements are being squished
Fix: add vertical scrolling
Make sure that when opening the selector you arye being scrolled to the currently selected engine

More questions/issues about engine selector:

* Is there a profile button in the engine selector? if there is, I cannot see it due to the above issue

### Engine matchmaking

The button opening the engine selector should take up a consistent amount of space
Layout inside the button:

* Left-aligned: Engine name
* right aligned: Board sizes

Below the button: Engine description

Next to the button, a more compact version of the Profile button - merge them into one element for visual clarity
