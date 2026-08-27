:- dynamic dirty/1.
:- dynamic vacuum_at/1.

room(kitchen).
room(hall).
room(bedroom).

connected(kitchen, hall).
connected(hall, bedroom).

dirty(kitchen).
dirty(bedroom).

vacuum_at(kitchen).

action(clean) :-
    vacuum_at(Place),
    dirty(Place).

action(move(hall)) :-
    vacuum_at(kitchen).

action(move(bedroom)) :-
    vacuum_at(hall).

action(stop) :-
    \+ dirty(_).

perform(clean) :-
    vacuum_at(Place),
    retract(dirty(Place)),
    format("Cleaned ~w~n", [Place]).

perform(move(Next)) :-
    retract(vacuum_at(_)),
    assert(vacuum_at(Next)),
    format("Moved to ~w~n", [Next]).

perform(stop) :-
    format("All rooms are clean. Vacuum stopped.~n", []).

run :-
    action(Action),
    perform(Action),
    Action \= stop,
    run.
run.

start :-
    run.
