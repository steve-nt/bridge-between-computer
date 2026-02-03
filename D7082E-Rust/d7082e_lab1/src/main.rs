//! A simple number guessing game in Rust.
//!
//! The program generates a random number between 1 and 100,
//! and prompts the player to guess it. After each guess, the program
//! provides feedback on whether the guess was too low, too high, or correct.

use std::cmp::Ordering;
use std::io;

use rand::Rng;

<<<<<<< HEAD
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
=======
/// Prompt the user for a guess and returns it as a u32.
/// 
/// This function continuously reads input from standard input until a valid positive
/// integer is provided. If invalid input is received, an error message is displayed and 
/// the user is re-prompted.
/// 
fn get_user_guess() -> u32 {
    loop {
        println!("Please input your guess (a positive number):");
>>>>>>> d647e9da11e09cc6b2309f7e2400d11efe15fcde

        let mut guess = String::new();

        io::stdin()
            .read_line(&mut guess)
            .expect("Failed to read line");

<<<<<<< HEAD
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
=======
        match guess.trim().parse::<u32>() {
            Ok(num) if num >= 1 && num <= 100 => return num,
            _ => println!("Invalid input. Please enter a number between 1 and 100."),
        }
    }
}

/// The entry point of the Number Guessing Game program.
/// 
/// Initializes the game by generating a random secret number between 1 and 100,
/// then enters a loop where the player makes guesses until they correctly identify 
/// the number. Provides feedback after each guess indicating if the guess was too low, 
/// too high, or correct. The game continues until the player guesses the correct number.
fn main() {
    println!("Welcome to the Number Guessing Game!");
    println!("I have selected a number between 1 and 100.");
    println!("Can you guess what it is?");

    let secret_number = rand::rng().random_range(1..=100);
>>>>>>> d647e9da11e09cc6b2309f7e2400d11efe15fcde

    loop {
        let guess = get_user_guess();

<<<<<<< HEAD
        println!("You guessed: {guess}");

        match guess.cmp(&secret_number) {
            Ordering::Less => println!("Too small!"),
            Ordering::Greater => println!("Too big!"),
            Ordering::Equal => {
                println!("You win!");
=======
        match guess.cmp(&secret_number) {
            Ordering::Less => println!("Too low! Try again."),
            Ordering::Greater => println!("Too high! Try again."),
            Ordering::Equal => {
                println!("Congratulations! You guessed the number {} correctly!", secret_number);
>>>>>>> d647e9da11e09cc6b2309f7e2400d11efe15fcde
                break;
            }
        }
    }
}
