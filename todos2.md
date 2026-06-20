# Next steps

The project is coming together.
Since its still in early early dev there is no need to migrate anything - the DB can be wiped until deployed in prod

## Submodules

I won't have a remote yet, if its possible to convert the repository into submodules without a remote go ahead

## Additional data

I want to store additional data that can be easily displayed on the frontend when needed
Important here are the fields and api endpoints
I want to keep everything as accessible and transparent as possible so track data and make it available as needed
However keep deduplicating info in mind. A users match data already contains their win ratio even if there is no percentage stored. These data points are for the frontend to compute and display.
Engines will also have that same data.

## Admin

Create a little admin interface (can be cli, tui, webinterface...) for administrators to manage settings like abandonment timers, engines and so on.
Try to think up an architecture that fits my modular concept and keeps attack surfaces minimal
For login administrators will use SSH-keys, in dev the Login button for admin will just let anyone in however in production you are only able to log in if your public key is in the environment or has been added to trusted keys in the admin interface

IMPORTANT: there should be no way for a regular user to access the admin login page just by clicking around until found

## Engines

Being able to register an engine like this is nice but a bit too open. For now keeping engine management privileged to administrator is a good solution that can be reverted later.

## More frontend

Incorporate an additional webpage (My stats, my data, ....?) into the frontend where user data and stats can be displayed.
Since there is no ownership checking for ID's yet add a way to check if the session is valid.
A User will now have their ID which identifies them in the database and a rotating session token which allows them to act as this user.
If no session token exists for the user the system issues a new ID and token, as if you were loged out and are now playing anonymously
In the future a successful login will issue a session token to the user that will be refreshed/rotated like discussed in a few lines above.

## Frontend bugs

By clicking around a little I discovered that sometimes buttons don't work.
I want you to take a look at the frontend and freshen it up a little, try to fix bugs.

## Other bugs

* Engines show as 'disconnected' in the match
