//! A simple number guessing game in Rust.
//!
//! The program generates a random number between 1 and 100,
//! and prompts the player to guess it. After each guess, the program
//! provides feedback on whether the guess was too low, too high, or correct.

use std::cmp::Ordering;
use std::io;

use rand::Rng;

/// Prompt the user for a guess and returns it as a u32.
/// 
/// This function continuously reads input from standard input until a valid positive
/// integer is provided. If invalid input is received, an error message is displayed and 
/// the user is re-prompted.
/// 
fn get_user_guess() -> u32 {
    loop {
        println!("Please input your guess (a positive number):");

        let mut guess = String::new();

        io::stdin()
            .read_line(&mut guess)
            .expect("Failed to read line");

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

    loop {
        let guess = get_user_guess();

        match guess.cmp(&secret_number) {
            Ordering::Less => println!("Too low! Try again."),
            Ordering::Greater => println!("Too high! Try again."),
            Ordering::Equal => {
                println!("Congratulations! You guessed the number {} correctly!", secret_number);
                break;
            }
        }
    }
}
