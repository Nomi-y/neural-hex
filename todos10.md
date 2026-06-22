# More touch ups

## UI

### Clock

When the clock is under a threshhold it shows miliseconds, i want a faster update time than the current 

### Matchmaking page

Remove the gradient from the buttons 
Remove the movement animation from the buttons as well

The cancel button should appear when 2 or more queues are active, current is 3+

When searching, display a 'Click to cancel' message at the bottom of each active button

The pill bar with selections 'Random - Invite link - Engine' should have the text for Random changed - something more close to the terms Matchmaking or Ranked

## Admin page

### Nav bar

Change the nav to a sidebar that can be toggled with a hamburger

### Login cookie

When successfully authenticating for the admin page set a cookie or issue a session token that keeps the login alive
Refreshing while the login is active keeps you logged in now

Timeout:
The login times out with inactivity, admin actions send a heartbeat that resets a configurable timer, default is 15 minutes
This means if the page is refreshed or reopened after the inactivity period expires the user is logged out
When the timeout passes while the site is open display a message and log the admin out automatically

Display the timer as a countdown in an easily readable location kept separate from nav or the page title

### Sign out

Since logins are sticky add a sign out button to the nav that clears logins

### Key selection

If possible to do securely, add an option in the browser to select and sign the challenge by selecting the private key in a file picker

