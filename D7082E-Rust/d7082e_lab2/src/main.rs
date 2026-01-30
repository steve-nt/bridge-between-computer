//! A number guessing game with high score tracking.
//!
//! Players guess a number between 1 and 10. After each correct guess, the game
//! records the player's name and number of guesses, displaying a sorted leaderboard.
//! Enter 0 to quit the game. High scores are automatically saved to disk.

use d7082e_lab2::{Leaderboard, Score};
use std::cmp::Ordering;
use std::io::{self, Write};
use rand::Rng;

const SECRET_RANGE_START: u32 = 1;
const SECRET_RANGE_END: u32 = 10;
const QUIT_CODE: u32 = 0;

/// Prompts the user for a text input.
///
/// Reads a line from stdin and returns it as a String (trimmed).
fn get_user_text_input(prompt: &str) -> String {
    loop {
        print!("{}", prompt);
        io::stdout().flush().expect("Failed to flush stdout");

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            println!("Failed to read input. Please try again.");
            continue;
        }

        let trimmed = input.trim().to_string();
        if !trimmed.is_empty() {
            return trimmed;
        }

        println!("Input cannot be empty. Please try again.");
    }
}

/// Prompts the user for a guess and validates it's within the game range.
///
/// # Returns
/// A valid guess in the range [1, 10] or 0 to quit
fn get_user_guess() -> u32 {
    loop {
        print!("Please input your guess (1-10, or 0 to quit): ");
        io::stdout().flush().expect("Failed to flush stdout");

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            println!("Failed to read input. Please try again.");
            continue;
        }

        match input.trim().parse::<u32>() {
            Ok(guess) => {
                if guess == QUIT_CODE || (guess >= SECRET_RANGE_START && guess <= SECRET_RANGE_END) {
                    return guess;
                }
                println!(
                    "Invalid input. Please enter a number between {} and {}, or 0 to quit.",
                    SECRET_RANGE_START, SECRET_RANGE_END
                );
            }
            Err(_) => println!("Invalid input. Please enter a valid number."),
        }
    }
}

/// Plays a single round of the guessing game.
///
/// Generates a random secret number and loops until the player guesses correctly.
///
/// # Returns
/// The number of guesses it took to win, or 0 if the player quit
fn play_round() -> u32 {
    let secret_number = rand::rng().random_range(SECRET_RANGE_START..=SECRET_RANGE_END);
    let mut guess_count = 0;

    loop {
        let guess = get_user_guess();

        if guess == QUIT_CODE {
            println!("Thanks for playing!");
            return 0;
        }

        guess_count += 1;

        match guess.cmp(&secret_number) {
            Ordering::Less => println!("Too low! Try again."),
            Ordering::Greater => println!("Too high! Try again."),
            Ordering::Equal => {
                println!(
                    "Congratulations! You guessed the number {} correctly in {} guesses!",
                    secret_number, guess_count
                );
                return guess_count;
            }
        }
    }
}

/// The main entry point of the guessing game application.
///
/// Initializes the game loop, loads existing high scores, and manages player sessions.
/// Players can play multiple rounds, and their scores are tracked and saved to disk.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔════════════════════════════════════╗");
    println!("║   Welcome to the Guessing Game!    ║");
    println!("╚════════════════════════════════════╝\n");
    println!("I have selected a number between {} and {}.", SECRET_RANGE_START, SECRET_RANGE_END);
    println!("Can you guess what it is?");
    println!("Enter 0 at any time to quit.\n");

    // Load existing leaderboard or create a new one
    let mut leaderboard = Leaderboard::load()
        .unwrap_or_else(|_| Leaderboard::new());

    loop {
        let guesses = play_round();

        if guesses == 0 {
            // Player quit
            break;
        }

        // Ask for player name
        let name = get_user_text_input("\nEnter your name: ");

        // Add score to leaderboard
        leaderboard.add_score(Score::new(name, guesses));

        // Save leaderboard to file
        leaderboard.save()
            .unwrap_or_else(|e| eprintln!("Warning: Could not save scores: {}", e));

        // Display leaderboard
        println!("{}", leaderboard.format_display());

        // Ask if player wants to play again
        println!("\nWould you like to play another round? (1 for yes, 0 for no)");
        let mut response = String::new();
        if io::stdin().read_line(&mut response).is_ok() {
            match response.trim() {
                "0" | "no" => break,
                _ => continue,
            }
        }
    }

    println!("\nThanks for playing! Your scores have been saved.");
    Ok(())
}

