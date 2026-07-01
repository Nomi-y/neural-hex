# Taining a neural network based engine

I want to create an engine that self trains to be pretty good at hex
I am familiar with neural networks, and fundamentals on LLMs so i have a couple thoughts already to go into this

## The engine

I want the engine to be an external engine following the documentation of ./Backend/Engines.md
If you did not update that document with evaluation engines yet do that before any other work
The end goal is to create an evaluation and play engine

## Built in features

1. Look a couple moves into the future
  - Has O(boardsize^n) scaling so we cant be generous with that.
  - Main goal is to solve endgame board states and to find wining paths
  - If finding winning paths can be precomputed then that is good
  - If precomputing is possible then combine these features as well

2. Engine should be able to recognize and play patterns (bridges, forks ...)

3. Other easy to implement features at your discretion

## Training

Train against itself with reward based feedback/evolution
I want to train this engine over a couple hours of real time compute so everything needed for training should be ready to scp to a VPS with more compute than my laptop
Instead of using the main stack, reuse code from the backend to create a self contained server instance

Before starting training, everything should be ready in a new directory in the project root

Make sure that training uses all of the resources available on the server once deployed

## Board size

For now lets train a fixed 13x13 board, this is large enough to not be computationally solved anywhere near in the future

## Play and evaluation

Engines output should be a list of next moves where temperature chooses the next move

## Thinking time

This is not a main concern but it should be not too large

