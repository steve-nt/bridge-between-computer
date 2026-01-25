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

## 🚀 Final Implementation (After All Refactoring)

Here's the final, refactored solution with documentation and improved error handling:

```rust
//! A simple number guessing game in Rust.
//!
//! The program generates a random number between 1 and 100,
//! and prompts the player to guess it. After each guess, the program
//! provides feedback on whether the guess was too low, too high, or correct.

use std::cmp::Ordering;
use std::io;

use rand::Rng;

/// Prompts the user for a guess and returns it as a u32.
///
/// This function continuously reads input from standard input until
/// a valid positive integer is provided. If invalid input is received,
/// an error message is displayed and the user is re-prompted.
///
/// # Returns
///
/// A `u32` representing the user's valid guess.
///
/// # Panics
///
/// Panics if reading from standard input fails.
fn get_user_guess() -> u32 {
    loop {
        println!("Please input your guess.");

        let mut guess = String::new();

        io::stdin()
            .read_line(&mut guess)
            .expect("Failed to read line");

        let guess: u32 = match guess.trim().parse() {
            Ok(num) => num,
            Err(_) => {
                println!("Please enter a valid number!");
                continue;
            }
        };

        return guess;
    }
}

fn main() {
    println!("Guess the number!");

    let secret_number = rand::thread_rng().gen_range(1..=100);

    loop {
        let guess = get_user_guess();

        println!("You guessed: {guess}");

        match guess.cmp(&secret_number) {
            Ordering::Less => println!("Too small!"),
            Ordering::Greater => println!("Too big!"),
            Ordering::Equal => {
                println!("You win!");
                break;
            }
        }
    }
}
```

### Key Improvements in Final Version:
1. **Extracted Function**: `get_user_guess()` handles all input validation and looping
2. **Crate-Level Docs**: `//!` explains what the entire program does
3. **Function Docs**: `///` documents the `get_user_guess()` function with Returns and Panics sections
4. **Better Error Handling**: Invalid input now displays `"Please enter a valid number!"` message
5. **Cleaner Main**: `main()` focuses on game logic, not input details
6. **No Compiler Warnings**: Code compiles cleanly with `cargo build`

---

## 🔄 Refactoring Improvements (Task 4)

### Before Refactoring:
```rust
// All input handling was in main()
loop {
    println!("Please input your guess.");
    let mut guess = String::new();
    io::stdin().read_line(&mut guess).expect("Failed to read line");
    let guess: u32 = match guess.trim().parse() {
        Ok(num) => num,
        Err(_) => continue,  // Silent failure
    };
    // ... rest of game logic
}
```

**Problems:**
- `main()` function is cluttered
- Input logic is repeated (or would be if refactored elsewhere)
- Invalid input silently fails with no user feedback
- Hard to test input logic separately

### After Refactoring:
```rust
/// Prompts the user for a guess and returns it as a u32.
/// ... (with full documentation)
fn get_user_guess() -> u32 {
    loop {
        println!("Please input your guess.");
        let mut guess = String::new();
        io::stdin().read_line(&mut guess).expect("Failed to read line");
        
        let guess: u32 = match guess.trim().parse() {
            Ok(num) => num,
            Err(_) => {
                println!("Please enter a valid number!");  // Clear feedback!
                continue;
            }
        };
        return guess;
    }
}

fn main() {
    println!("Guess the number!");
    let secret_number = rand::thread_rng().gen_range(1..=100);
    
    loop {
        let guess = get_user_guess();  // Clean and simple!
        println!("You guessed: {guess}");
        match guess.cmp(&secret_number) {
            // ... comparison logic
        }
    }
}
```

**Benefits:**
- ✅ Separation of concerns: input handling is separate from game logic
- ✅ Better error feedback: users know what went wrong
- ✅ More testable: input function can be tested independently
- ✅ More readable: `main()` is clear and focused
- ✅ Documentation: function purpose is clearly documented

---

## 📝 Documentation Added

### Crate-Level Documentation (`//!`)
Located at the top of `src/main.rs`, explains the purpose of the entire program:
```rust
//! A simple number guessing game in Rust.
//!
//! The program generates a random number between 1 and 100,
//! and prompts the player to guess it. After each guess, the program
//! provides feedback on whether the guess was too low, too high, or correct.
```

### Function Documentation (`///`)
Documents the `get_user_guess()` function with standard Rust doc format:
```rust
/// Prompts the user for a guess and returns it as a u32.
///
/// This function continuously reads input from standard input until
/// a valid positive integer is provided. If invalid input is received,
/// an error message is displayed and the user is re-prompted.
///
/// # Returns
///
/// A `u32` representing the user's valid guess.
///
/// # Panics
///
/// Panics if reading from standard input fails.
```

**View the Generated Documentation:**
```bash
cargo doc --open
```
This generates HTML documentation and opens it in your browser!

---

## ✨ Error Handling Improvements (Bonus Feature)

### Original Approach (Tutorial):
```rust
let guess: u32 = match guess.trim().parse() {
    Ok(num) => num,
    Err(_) => continue,  // Silent - user doesn't know why it failed
};
```

### Improved Approach (Our Implementation):
```rust
let guess: u32 = match guess.trim().parse() {
    Ok(num) => num,
    Err(_) => {
        println!("Please enter a valid number!");  // User-friendly feedback
        continue;
    }
};
```

**Example Session:**
```
Guess the number!
Please input your guess.
abc
Please enter a valid number!        ← Clear feedback!
Please input your guess.
50
You guessed: 50
Too big!
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

### Mistake 2: Not Handling the Result from parse()
```rust
// ❌ Wrong - will not compile
let guess: u32 = input.trim().parse();

// ✅ Right - using expect() (simple, crashes on error)
let guess: u32 = input.trim().parse().expect("Please type a number!");

// ✅ Better - using match (graceful error handling)
let guess: u32 = match input.trim().parse() {
    Ok(num) => num,
    Err(_) => {
        println!("Please enter a valid number!");
        continue;  // Ask for input again
    }
};
```

### Mistake 3: Using `u32` for values that could be negative
```rust
// If you want to support negative numbers, use i32 instead
let guess: i32 = input.trim().parse().expect("Please type a number!");
```

### Mistake 4: Forgetting to call `.trim()`
```rust
// ❌ Problem: "42\n" won't parse as a number (the \n newline causes issues)
let guess: u32 = input.parse().expect("Please type a number!");

// ✅ Solution: Remove the newline and spaces first
let guess: u32 = input.trim().parse().expect("Please type a number!");
```

### Mistake 5: Not Providing User Feedback for Invalid Input
```rust
// ❌ Not user-friendly - silent failure
let guess: u32 = match input.trim().parse() {
    Ok(num) => num,
    Err(_) => continue,
};

// ✅ User-friendly - explains what went wrong
let guess: u32 = match input.trim().parse() {
    Ok(num) => num,
    Err(_) => {
        println!("Please enter a valid number!");
        continue;
    }
};
```



---

## 🎓 Lab Tasks Completion Status

### ✅ Task 1: Setting up Rust
- [x] Installed Rust and setup git environment
- [x] Setup VSCode with Rust Analyzer support
- [x] Created user on Vesuvio
- [x] Forked lab on Vesuvio
- [x] Cloned repo locally
- [x] Completed guessing game tutorial

### ✅ Task 2: Read the Docs
- [x] Added documentation links to README.md for:
  - [String](https://doc.rust-lang.org/std/string/struct.String.html)
  - [read_line](https://doc.rust-lang.org/std/io/trait.BufRead.html#method.read_line)
  - [Result](https://doc.rust-lang.org/std/result/enum.Result.html)
  - [println!](https://doc.rust-lang.org/std/macro.println.html)
  - [match](https://doc.rust-lang.org/std/keyword.match.html)
  - [break](https://doc.rust-lang.org/std/keyword.break.html)

### ✅ Task 3: External Dependencies
- [x] Verified `rand` crate documentation
- [x] Documented both APIs:
  - [gen_range](https://docs.rs/rand/latest/rand/trait.Rng.html#method.gen_range) (current stable API in 0.8.5)
  - [random_range](https://docs.rs/rand/latest/rand/trait.Rng.html#method.random_range) (future API)
- [x] Code compiles without warnings
- [x] Using correct API for rand 0.8.5

### ✅ Task 4: Refactoring and Local Documentation
- [x] Extracted user input into `get_user_guess()` function
- [x] Added crate-level documentation (`//!`)
- [x] Added function documentation (`///`)
- [x] Generated HTML documentation with `cargo doc`
- [x] **BONUS**: Added error feedback for invalid input

### ⏳ Task 5: Submit, Review, and Finalize
- [ ] Commit/push code to Vesuvio
- [ ] Submit repository link in Canvas
- [ ] Review 2 peer repositories
- [ ] Create GitHub issues for peer reviews
- [ ] Fix issues based on feedback
- [ ] Write learning reflection

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

