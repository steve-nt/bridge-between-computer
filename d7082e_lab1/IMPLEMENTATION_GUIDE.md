# D7082E Lab 1 - Rust Guessing Game Implementation Guide

## Welcome to Rust! 🦀

Hey there! If you're reading this, you're about to learn one of the most powerful programming languages out there. Rust is used by companies like Mozilla, Microsoft, and Amazon to build fast, safe, and reliable software. Let's dive in!

---

## 📋 Overview of the Assignment

Your mission is to build a **Guessing Game** in Rust. Here's what it does:

1. The computer picks a random number between 1 and 100
2. You try to guess it
3. After each guess, the program tells you: "Too small", "Too big", or "You got it!"
4. You keep guessing until you get it right

Think of it like a mind-reading game where the computer thinks of a number and you have to figure it out!

---

## 🎯 Key Concepts You'll Learn

### 1. **Variables and Mutability** (Like boxes that hold things)
In Rust, variables are like labeled boxes that store data. But here's the twist:
- By default, variables are **immutable** (can't change)
- To make them changeable, you add `mut` (like putting a "modifiable" sticker on the box)

```rust
let number = 5;        // Can't change this
number = 10;           // ERROR! 🚫

let mut guess = 5;     // Can change this
guess = 10;            // OK! ✅
```

**Why?** This forces you to think carefully about what can change. It prevents bugs!

**Search keyword:** "Rust mutability"

---

### 2. **The `Result` Type** (Like getting a success or error message)
Many Rust operations can either succeed or fail. Instead of crashing, Rust returns a `Result`:
- `Ok(value)` = Success! Here's your data
- `Err(error)` = Oops! Something went wrong

Think of it like a text message that either says "I got your order ✅" or "Sorry, we're out of stock ❌"

**Search keyword:** "Rust Result enum"

---

### 3. **Pattern Matching with `match`** (Like a smart if/else for Rust)
Instead of writing multiple `if/else` statements, Rust uses `match` to handle different cases:

```rust
match result {
    Ok(value) => println!("Success: {}", value),
    Err(error) => println!("Error: {}", error),
}
```

It's like a vending machine with different buttons—each button triggers a different action!

**Search keyword:** "Rust match expression"

---

### 4. **Loops** (Keep doing something until you're done)
The guessing game needs to keep asking for guesses until the player wins. We use a `loop`:

```rust
loop {
    // Ask for a guess
    // Check if they won
    // If yes, break out of the loop
}
```

**Search keyword:** "Rust loop, break statement"

---

### 5. **The `rand` Crate** (Using someone else's code)
A "crate" is a library of code someone wrote and shared. The `rand` crate generates random numbers so we don't have to write that ourselves.

**Search keywords:** "Rust crates.io", "Rust dependencies Cargo.toml"

---

### 6. **String vs &str** (Text storage)
- `String` = owned text (like your own notebook)
- `&str` = borrowed text (like reading someone else's notebook)

For this project, `String` is what we need when we read user input.

**Search keyword:** "Rust String vs &str"

---

## 🛠️ Step-by-Step Explanation

### Step 1: Import what we need
```rust
use std::io;
use std::cmp::Ordering;
use rand::Rng;
```
- `io` = input/output (to read user input)
- `Ordering` = tells us if a number is bigger, smaller, or equal
- `rand::Rng` = the random number generator

### Step 2: Set up the program
```rust
fn main() {
    // Your code goes here
}
```
`main` is where Rust starts running your program. It's like the starting line of a race!

### Step 3: Generate a random number
```rust
let secret_number = rand::thread_rng().gen_range(1..=100);
```
This creates a random number between 1 and 100 (including 100). The `thread_rng()` part means "use this thread's random number generator."

### Step 4: The main game loop
```rust
loop {
    // Ask user for guess
    // Read their input
    // Parse it to a number
    // Compare with secret number
    // Tell them if they won
}
```

### Step 5: Read user input
```rust
let mut input = String::new();
io::stdin().read_line(&mut input).expect("Failed to read line");
```
- `String::new()` = create an empty text box
- `read_line()` = wait for user to type something
- `&mut input` = give the function permission to modify (write to) our text box
- `.expect()` = if something goes wrong, show an error message

### Step 6: Parse and handle errors
```rust
let guess: u32 = input.trim().parse().expect("Please type a number!");
```
- `.trim()` = remove extra spaces
- `.parse()` = convert text "42" into the number 42
- This returns a `Result`, and `.expect()` handles the error

### Step 7: Compare and respond
```rust
match guess.cmp(&secret_number) {
    Ordering::Less => println!("Too small!"),
    Ordering::Greater => println!("Too big!"),
    Ordering::Equal => {
        println!("You won!");
        break;  // Exit the loop
    }
}
```

---

## 📚 Resources for Deeper Learning

1. **The Rust Book (Official Guide)**
   - Chapter 2: Programming a Guessing Game
   - URL: https://doc.rust-lang.org/book/ch02-00-guessing-game-tutorial.html

2. **Rust by Example (Interactive Learning)**
   - URL: https://doc.rust-lang.org/rust-by-example/

3. **The Rust Standard Library Documentation**
   - URL: https://doc.rust-lang.org/std/

4. **Crates.io (Finding Libraries)**
   - URL: https://crates.io/
   - Search for "rand" to see the random number generator

5. **Rust Documentation: Common Programming Concepts**
   - Variables: https://doc.rust-lang.org/book/ch03-01-variables-and-mutability.html
   - Data Types: https://doc.rust-lang.org/book/ch03-02-data-types.html
   - Functions: https://doc.rust-lang.org/book/ch03-03-how-functions-work.html

---

## 🚀 Complete Code Implementation

Here's the full, working solution with detailed comments:

```rust
//! A simple guessing game where the player tries to guess a random number
//! between 1 and 100. The computer tells you if your guess is too high,
//! too low, or correct.
//!
//! This is a beginner-friendly program that teaches:
//! - Reading user input
//! - Pattern matching
//! - Loops
//! - The Result type
//! - Using external crates (libraries)

use std::cmp::Ordering;
use std::io;
use rand::Rng;

fn main() {
    // Display a welcome message
    println!("Welcome to the Guessing Game! 🎮");
    println!("I'm thinking of a number between 1 and 100.");
    println!("Can you guess it?\n");

    // Generate a random secret number between 1 and 100 (inclusive)
    // rand::thread_rng() = get a random number generator for this thread
    // gen_range(1..=100) = generate a number from 1 to 100 (the = means including 100)
    let secret_number = rand::thread_rng().gen_range(1..=100);

    // Keep track of how many guesses the player has made
    // We use 'mut' because we'll be incrementing this value
    let mut guess_count = 0;

    // Start an infinite loop - we'll break out when the player guesses correctly
    loop {
        // Increment the guess counter
        guess_count += 1;

        // Ask the user for their guess
        println!("Please input your guess (Guess #{}): ", guess_count);

        // Create a new empty String to store the user's input
        // String::new() = create an empty, modifiable text container
        let mut input = String::new();

        // Read a line from standard input (the keyboard)
        // stdin() = get access to the keyboard input
        // read_line(&mut input) = read what the user types and store it in 'input'
        //                         &mut means "let me modify this variable"
        // expect("...") = if something goes wrong, show this error message and crash
        //                 (This is fine for learning - in real code we'd handle it better)
        io::stdin()
            .read_line(&mut input)
            .expect("Failed to read line");

        // Convert the text input into a number
        // .trim() removes whitespace (spaces, newlines) from the edges
        //         This is important because read_line() includes the newline character!
        // .parse() tries to convert the text into the type we specify (u32 = unsigned 32-bit integer)
        //          This returns a Result: either Ok(number) or Err(error)
        // expect("...") = if parse fails, show this message
        //
        // We store it in a variable called 'guess' of type u32
        let guess: u32 = input
            .trim()
            .parse()
            .expect("Please type a valid number!");

        // Display the guess they made
        println!("You guessed: {}\n", guess);

        // Compare the guess with the secret number
        // .cmp() = "compare" - returns an Ordering (Less, Greater, or Equal)
        // match = look at the result and handle each possible case
        match guess.cmp(&secret_number) {
            // Case 1: The guess is less than the secret number
            Ordering::Less => println!("📉 Too small! Try a bigger number.\n"),

            // Case 2: The guess is greater than the secret number
            Ordering::Greater => println!("📈 Too big! Try a smaller number.\n"),

            // Case 3: The guess equals the secret number - they won!
            Ordering::Equal => {
                println!("🎉 You got it! The number was {}!", secret_number);
                println!("It took you {} guesses to win!", guess_count);
                break;  // Exit the loop - the game is over!
            }
        }
    }

    println!("\nThanks for playing! 👋");
}
```

---

## ✅ How to Run Your Code

1. **Setup** (first time only):
   ```bash
   # Make sure you're in the lab1 directory
   cd /home/steven/Desktop/d7082e_lab1
   
   # Check that dependencies are set up
   cargo build
   ```

2. **Run the game**:
   ```bash
   cargo run
   ```

3. **Clean up** (optional):
   ```bash
   cargo clean  # Removes build files
   ```

---

## 🐛 Common Mistakes and How to Fix Them

### Mistake 1: Forgetting `mut` when you need to change a variable
```rust
// ❌ Wrong
let input = String::new();
io::stdin().read_line(&input);  // Error! input is immutable

// ✅ Right
let mut input = String::new();
io::stdin().read_line(&mut input);  // OK!
```

### Mistake 2: Forgetting to `.expect()` on a Result
```rust
// ❌ Wrong
let guess: u32 = input.trim().parse();  // Error! parse() returns a Result

// ✅ Right
let guess: u32 = input.trim().parse().expect("Please type a number!");
```

### Mistake 3: Using `u32` for a value that should be negative
```rust
// If you want to support negative numbers, use i32 instead
let guess: i32 = input.trim().parse().expect("Please type a number!");
```

### Mistake 4: Forgetting to call `.trim()`
```rust
// ❌ Problem: "42\n" won't parse as a number (the \n newline causes issues)
let guess: u32 = input.parse().expect("Please type a number!");

// ✅ Solution: Remove the newline first
let guess: u32 = input.trim().parse().expect("Please type a number!");
```

---

## 🎓 Next Steps (After Completing This)

1. **Task 2**: Add documentation links to your README
2. **Task 3**: Update the `rand` crate API (it's changing from `gen_range()` to `gen_range()` with new syntax)
3. **Task 4**: Refactor by extracting the input-reading code into its own function
4. **Add better error handling**: Instead of `.expect()`, use `match` to handle errors gracefully
5. **Add validation**: Make sure guesses are between 1 and 100

---

## 📖 Helpful Search Terms for Learning More

- "Rust ownership"
- "Rust borrowing and references"
- "Rust trait bounds"
- "Rust error handling best practices"
- "Rust documentation comments"
- "How to use cargo"
- "Rust semicolons vs no semicolons"
- "Rust type inference"

---

## 🤝 Need Help?

- **Compiler errors?** Read them carefully! Rust's error messages are usually very helpful.
- **Confused about a concept?** Search the Rust Book chapters - they explain everything well.
- **Want to explore further?** Try modifying the game (e.g., add difficulty levels, track statistics).

---

## Remember

**You're learning one of the most respected programming languages in the world.** It might feel challenging at first, but every line you write teaches you valuable lessons about:
- Memory safety (no random crashes!)
- Concurrency (running many things at once safely)
- Speed (Rust is as fast as C/C++)
- Expressiveness (you can write very clear, readable code)

Take your time, understand the concepts, and you'll be writing amazing Rust code in no time! 🚀

---

**Happy coding! Feel free to experiment and break things—that's how you learn.** 🦀

