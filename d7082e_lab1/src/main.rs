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
