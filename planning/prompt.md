# Creating an online game application

a copy of this prompt is in ./prompt.md

I want to create an online application to play the 2-player board game 'Hex'

## Architecture & Tech stack

Following are my base thoughts going into this project

* I want to separate backend/frontendend layers
* For ease of further development these should be in their separate subdirectories and even git repositories
* As programming language use Typescript with Bun.
* I am thinking about trying out spacetimedb, if you think a different database would be more suited for this project you have the go-ahead
* Deployment of both backend and frontend servers will be with docker using multi-stage builds

## Writing Code

When writing code follow the following conventions:

* PascalCase for:
  * File names
  * Directory names
  * Functions, classes, methods, class members
* Keep comments at an absolute minimum, only explain genuinely confusing parts
* Make sure any part of code is maintainable, any feature should be easy to expand on, add on, scale up or rework
* Write sensible unit tests for features that need them. important here is testing logic that might be flawed - code coverage is irrelevant.
* Coding Conventions such as DRY should be respected only when sensible. Bloating code is a greater sin.
* Lastly, keep the logic in the code simple. You can use advanced TypeScript features but the actual logic should be easy to understand just by reading line by line

## Permissions

The following is a list of permissions you are granted for this project

* You are allowed to install Bun packages as needed, try keeping them on the lower end however
* You are allowed to run git commands, so set up local repositories and commit on feature/checkpoint. Separate branches are not needed as of right now.

## Features

These are the features I want you to implement from the perspective of the end user:

* Play a game of hex, this includes
  * Random match against anyone searching
  * Match someone with an invite link
* Game functionality will follow the standard rules of hex with a couple customisation options, those being
  * Clock settings
    * Unlimited
    * X+Y time format, for example 5 minutes + 10s every move, set the boundries for these to reasonable numbers
  * Board Size, once again sensible boundries and limits
* When refreshing you will rejoin an ongoing match
* Match limit of 1, you cannot have multiple matches at the same time

## Out-of-Scope

These points are clearly out of scope for the basic implementation

* An engine to play against
* Game analysis/evaluation
* Any ranking system
* Any user system or other login based system


## Client

Choose how to implement the client based on the needs of an online turn-based game.

## Server

Program the server based on the discussed backend architecture.
The server will orchestrate games and interact with the client over either REST or websockets, depending on the requested feature.

## Output

When making important design decisions try acting yourself, by keeping in mind that an online game should be scalable.

a copy of this prompt is in ./prompt.md
