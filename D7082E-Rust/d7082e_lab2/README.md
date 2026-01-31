# D7082E - Lab 2

Covers lectures 3 and 4, (chapters 7-12 of the [Rust Book](https://doc.rust-lang.org/book/)).

## Task 1 - Setting up the repository

In this lab you will extend the guessing game that you implemented in lab1.

- Fork the lab on Vesuvio.

- Clone the repo to your local machine.

- Copying the `main.rs` source file from your lab1 and add the needed dependency to your `Cargo.toml`.

- Test that the application runs as expected.

- Make a first commit and push it.

## Task 2 - High Score

- Reduce the range of random numbers to make it easier to guess the secret, e.g., to 1..10.

- Change the program so that it loops, until the user decides to quit, e.g., you can use e.g., 0 as input for exiting the application. Check the Rust book chapter 12.3 for process exit, the application should return with the error code 0, to indicate success.

- Keep track of the number of guesses, and once the users "wins":

  - Print the number of guesses it took.

  - Reset the counter and generate a new secret to solve.
  - Ask the user for his name.
    - Use a collection to store name and number of guesses.
  - Print the collection sorted such that the lowest number of guesses is printed first.

  The printout should look something like:

  ```txt
    All time bests scores!
     #   Name
    --   --------
     1   Anki
     3   Kalle
     4   Kalle
     4   Kalle
     4   Olle
    13   Arne
  ```

  Notice, the same name can occur several times. For cases where the same score occurs, the latest players score and name should be listed first. (So in this case Olle scored 4, then Kalle scored 4 two times). The scores (number of turns needed) should be right formatted, while the name should be left formatted. We assume that number of turns needed is no more than 99, thus two characters is sufficient.

- Implement the score board functionality in `lib.rs` (or a sub module to the library).

- Make unit tests to show that edge cases (including the one discussed above) works as expected.

- Commit and push your solution.

## Task 3 - High Score to File

In this task you will store the high store collection to file, e.g., `high_score.txt`.

- The storage format should be documented, and humanly readable. Besides that, you can decide what you think is best/easiest.

- Once the application starts it should look for a high score file, e.g., in current folder.

  - If existing it should load the collection, else create a new empty high score collection.

- After each "win" the high score file should be written to file.

- Commit and push your solution.

- Hint, you may use any third party library, e.g., `serde` which facilitates serialization and de-serialization of (custom) data structures.

  Use a test driven approach to implement the file handling, else you will get really bored guessing numbers. So start by designing tests, for the serialization and deserialization of your high score collection. Then integrate file handling to the application when it works.

- Commit and push your code.

## Task 4 - Error Handling

- Look over your code and make sure that no operation may panic (use `Result` and/or the `Error` trait where applicable).

- For fatal, unrecoverable errors it is ok to panic (e.g., by `expect`). For each such case remaining in your code, make an explicit comment that motivates why the error is un-recoverable.

- Commit and push your code.

## Task 5 - Documentation

- Document the library API and internals as well as the application code, such that it becomes easy to understand both for the reviewer and the end user (player).

- Commit and push your solution.

## Task 6 - Submit, Review and Finalize

Business as usual, same procedure as for Lab 1.

## Learning Outcomes

- Reflect in your own words what you have learned by doing this lab. Hint, compare your knowledge regarding Rust, tooling etc., before and after doing this lab.

< Your Text Here>

## Congrats

You have now taken the second step to robust and reliable programming in Rust.

Now you can now write non-trivial applications based on a given specification. Later in the course, we will further extend on the guessing game using more advanced language features, external crates etc.
