use std::cmp::Ordering;
use std::io;

use rand::Rng;

fn main() {
    // Display welcome message
    println!("Guess the number!");

    // Generate a random secret number between 1 and 100
    let secret_number = rand::thread_rng().gen_range(1..=100);

    // Loop until the player guesses correctly
    loop {
        println!("Please input your guess.");

        // Create a mutable string to store user input
        let mut guess = String::new();

        // Read input from standard input
        io::stdin()
            .read_line(&mut guess)
            .expect("Failed to read line");

        // Convert the string input to u32, handling invalid input
        let guess: u32 = match guess.trim().parse() {
            Ok(num) => num,
            Err(_) => continue,
        };

        // Display the user's guess
        println!("You guessed: {guess}");

        // Compare the guess with the secret number and provide feedback
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
