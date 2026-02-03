# D7082E - Lab 1

Covers lectures 1 and 2 (chapters 1-6 of the [Rust Book](https://doc.rust-lang.org/book/)).

<<<<<<< HEAD
You may collaborate with other students for solving the Tasks, but you will subimit review etc, individually.
=======
You may collaborate with other students for solving the Tasks, but you will submit review etc, individually.
>>>>>>> d647e9da11e09cc6b2309f7e2400d11efe15fcde

If you have collaborated with others. Change the below to match the way you collaborated. In this way, suspection of plagiarism can be avoided even if your repos turn out to be very similar.

- [](), Mr X, came up with the idea of Y which I used in my code
<<<<<<< HEAD
- [](), Mrs Z, suggest using R, which turned ut great, so I used that as well.
=======
- [](), Mrs Z, suggest using R, which turned out great, so I used that as well.
>>>>>>> d647e9da11e09cc6b2309f7e2400d11efe15fcde
- Etc.

## Task 1 - Setting up Rust

- Install Rust and setup a working git environment. WSL recommended for Windows users.

- Setup `vscode` or editor of choice with `rust analyzer` support (or similar).

- Create a user on Vesuvio. (This you have already done.)

- Fork the lab on Vesuvio.

- Clone the repo to your local machine.

Complete the [Programming a Guessing Game](https://doc.rust-lang.org/book/ch02-00-guessing-game-tutorial.html) tutorial. Notice, You should replace the provided `src/main.rs` with Your guessing game code, do not create a new Rust crate.
<<<<<<< HEAD
You should run the following command to add the rand crate
cargo add rand 
=======
>>>>>>> d647e9da11e09cc6b2309f7e2400d11efe15fcde

The program should/will compile and run but will generate a warning, You will fix that later.

## Task 2 - Read the docs

Use the Rust docs [std](https://doc.rust-lang.org/stable/std/index.html) to find the documentation for the types used. Update this file (`README.md`), with direct links to the corresponding documentation. Make sure the links get you where intended when clicked.

- [String](https://doc.rust-lang.org/std/string/struct.String.html)
- [read_line](https://doc.rust-lang.org/std/io/trait.BufRead.html#method.read_line)
- [Result](https://doc.rust-lang.org/std/result/enum.Result.html)
- [println!](https://doc.rust-lang.org/std/macro.println.html)
- [match](https://doc.rust-lang.org/std/keyword.match.html)
- [break](https://doc.rust-lang.org/std/keyword.break.html)

Navigating and understanding the docs is _instrumental_ to programming in Rust. Using Rust analyzer or similar tools, help you when programming. You may optionally use an AI assistant (LLM) for helping you write, document and test your code.

## Task 3 - External dependencies

Follow the link from the official crates.io release for the [rand](https://crates.io/crates/rand) to find its documentation. Update the link below, and verify that it gets you where intended when clicked.

- [gen_range](https://docs.rs/rand/latest/rand/trait.Rng.html#method.gen_range)

As you see the API is about to change, and the function is marked as `Deprecated`. The docs suggest the new (and improved) API. Change the code accordingly such that it compiles and runs without warnings.

Update the link to the new API used below.

- [random_range](https://docs.rs/rand/latest/rand/trait.Rng.html#method.random_range)

## Task 4 - Re-factoring and Local documentation

First re-factor your code, to provide a function that returns user input as a `u32`. (It should loop until valid input data is received.)

Secondly, update the source code with suitable documentation, `//!` for crate level docs and `///` per item (e.g., data structure, function etc.). Since its a simple application, documentation will be trivial, so its mostly for learning the documentation process.

Run:

`cargo doc --open` to verify that the documentation is created correctly.

## Task 5 - Submit, Review and Finalize

- First commit/push your code to Vesuvio.

- In canvas assignment, submit a link to your repository (make sure its visible to others).

- Once assigned two projects to review follow the review assignment in Canvas, and create one git issue per reviewed repository.

- In canvas assignment, submit links to your created issues (on Vesuvio).

- Once you received reviews, update your repository accordingly.

- Once the issues you have created have been attended to, you can close the issues on Vesuvio.

## Learning Outcomes

- Reflect in your own words what you have learned by doing this lab. Hint, compare your knowledge regarding Rust, tooling etc., before and after doing this lab.

<<<<<<< HEAD
D7082ELAB1 Reflection 1 

D7082ELAB1 Reflection 2

D7082ELAB1 Reflection 3

D7082ELAB1 Reflection 4

D7082ELAB1 Reflection 5
=======
- **D7082ELAB1 Reflection 1**: Installing Rust and verifing the installation is working correctly using the Rust compiler and Cargo package manager. Since, I have not worked before with Rust everything was new.

- **D7082ELAB1 Reflection 2**: Understanding Immutability, I have not seen the concept of mutable and immutable variables in Rust and in other languages, since I do not develop software.

- **D7082ELAB1 Reflection 3**: Write and use match expressions to handle multiple possible outcomes and make decisions based on different values or patterns. I have seen the if else and the case expressions.

- **D7082ELAB1 Reflection 4**: Use the cmp() method and Ordering enum to compare numeric values

- **D7082ELAB1 Reflection 5**: The use of external external Rust libraries (crates) and the concept of them, and the differences with other languages like python and golang

- **D7082ELAB1 Reflection 6**: The different documentation standards, writing crate-level documentation using //! comments and function-level documentation using /// comments following Rust conventions. And creating documentation running the `cargo doc --open` command
>>>>>>> d647e9da11e09cc6b2309f7e2400d11efe15fcde

- In canvas assignment submit link to your final lab1 repository.

## Congrats

You have now completed the first steps to learning a modern systems level programming language and you have mastered the tooling involved to manage a Rust project!
