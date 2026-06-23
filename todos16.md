# Backend behaviour

./project-rules.md

This directive is a list of backend changes. You will be working in parallel to an agent that works on the frontend while you work on this list.

## Engine resigning

I overlooked the fact that engines are allowed to resign which lead to a hilarious incident while testing.
This is due to an intended behaviour of deleting any match against an engine if it was timed, resigned or abandoned.
From now on keep these matches in the match history

## External engine network performance

To get an overview look at the external engines section in ./Backend/Engines.md
I am concerned about network performance if an exeternal engine has many matches at once
Troubleshoot this issue and implement a fix
since this is quite complicated create an external engine in a new directory ./ExternalEngines for testing

Engine credentials
Token: 3b71385db4704d8584b441f274ff2bc7ea4dbce0846e41e99f921a292a1e9acf 
EngineId: d05ae246-b110-4973-95ce-bc7730215ca4

After implementing and verifying your fix update the engine documentation as well

## Managed engines

Managed engines have become async for performance reasons
If this is not documented, add this behaviour to the manage engine documentation
