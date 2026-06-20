# Large feature plans

Now that everything works adding users is the next step
A user is just a unique username and password.
Anyone will also just be able to play anonymously with no issues.

## Out of scope

For now email verification, password resets and so on are all out of scope
Password checking is also out of scope, any reasonable input is a valid password, no forced length, numbers, symbals etc
The username should only be cheked for duplicates

## Features for user

A registered User will have the following features:

* Instead of the userId (which is shown when not logged in) the username is shown to self and others
* When rankings are added (later, out of scope for now) a user should be able to gain elo or a similar metric and access skill based matchmaking. for now just adding the ability to add those features later is enough
* Clicking on a username in your match history shows the user profile - basically 'My Stats' but for a different user
* A user has a registered at date which can be seen in the user profile

## UI Changes

Instead of a My-Stats Tab there should be a dedicated button to show the user profile and an option to log out in the user profile

The administrator UI should get an extra tab/section for user management, basically for now just the option to delete any user

## Password

 The user password will be stored securely - salted and hashed, this system should be production-ready
