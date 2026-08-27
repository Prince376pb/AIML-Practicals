:- dynamic preference/1.
:- dynamic recommended/1.

book(harry_potter, fantasy, beginner, fiction).
book(the_hobbit, fantasy, beginner, adventure).
book(alchemist, fiction, beginner, motivation).
book(atomic_habits, self_help, beginner, productivity).
book(deep_work, self_help, intermediate, productivity).
book(rich_dad_poor_dad, finance, beginner, business).
book(the_psychology_of_money, finance, intermediate, business).
book(introduction_to_algorithms, technology, advanced, computer_science).
book(clean_code, technology, intermediate, programming).
book(steve_jobs, biography, intermediate, technology).

get_preferences :-
    write('Enter your preferred genre: '),
    read(Genre),
    assert(preference(genre(Genre))),
    write('Enter your reading level: '),
    read(Level),
    assert(preference(level(Level))),
    write('Enter your preferred subject: '),
    read(Subject),
    assert(preference(subject(Subject))).

suitable(Book) :-
    book(Book, Genre, Level, Subject),
    preference(genre(Genre)),
    preference(level(Level)),
    preference(subject(Subject)),
    \+ recommended(Book).

action(recommend(Book)) :-
    suitable(Book).

action(stop) :-
    \+ suitable(_).

perform(recommend(Book)) :-
    assert(recommended(Book)),
    format('Recommended Book: ~w~n', [Book]).

perform(stop) :-
    write('No more suitable books found.'), nl.

start :-
    retractall(preference(_)),
    retractall(recommended(_)),
    get_preferences,
    recommend_books.

recommend_books :-
    action(Action),
    perform(Action),
    Action \= stop,
    recommend_books.

recommend_books :-
    action(stop),
    perform(stop).
