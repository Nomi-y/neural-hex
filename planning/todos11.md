# Clock logic and timeout + UI fix

This directive contains instructions to extending the clock logic

## Logic and settings

Now that there are various clock modes capping the minimum for baseMs at 1000 seems a little restrictive
Extend the clock logic and validation settings to allow setting BaseMS to 0 for the Delay setting
For validating a correct time baseMs + delayMs must be > 0
This change is both for the admin page and the match creation page

## Timeout bug

creating a 1s+1s Delay match and timing out leaves the game in a bugged state, no win/loss happens
This is probably some race condition

## Graphics

When limiting the board size to a small amount the number buttons becomes right-aligned - they should be centered
