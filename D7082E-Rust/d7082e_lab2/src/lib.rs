//! Scoring system for the guessing game.
//!
//! This module provides functionality to track, store, and manage high scores
//! for the guessing game. Scores are sorted by number of guesses (ascending),
//! with ties broken by insertion order (most recent first).

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fs;
use std::io;
use std::path::Path;

<<<<<<< HEAD
const SCORE_FILE: &str = "high_scores.txt";
=======
const SCORE_FILE: &str = "high_scores.json";
>>>>>>> 5aa402a546a7707f2cc5a517dfdfcc5968e19f10

/// Represents a single high score entry.
///
/// Contains the player's name and the number of guesses it took to win.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Score {
    /// The name of the player.
    pub name: String,
    /// The number of guesses it took to win.
    pub guesses: u32,
}

impl Score {
    /// Creates a new Score entry.
    ///
    /// # Arguments
    /// * `name` - The player's name
    /// * `guesses` - The number of guesses it took to win
    ///
    /// # Examples
    /// ```
    /// use d7082e_lab2::Score;
    /// let score = Score::new("Alice".to_string(), 5);
    /// assert_eq!(score.name, "Alice");
    /// assert_eq!(score.guesses, 5);
    /// ```
    pub fn new(name: String, guesses: u32) -> Self {
        Score { name, guesses }
    }
}

/// Manages the collection of high scores.
///
/// The leaderboard maintains scores in a list and provides functionality
/// to add new scores, sort them, and persist them to/from files.
#[derive(Debug, Serialize, Deserialize)]
pub struct Leaderboard {
    /// The list of scores, stored in insertion order internally.
    scores: Vec<Score>,
}

impl Leaderboard {
    /// Creates a new empty leaderboard.
    ///
    /// # Examples
    /// ```
    /// use d7082e_lab2::Leaderboard;
    /// let leaderboard = Leaderboard::new();
    /// assert_eq!(leaderboard.len(), 0);
    /// ```
    pub fn new() -> Self {
        Leaderboard {
            scores: Vec::new(),
        }
    }

    /// Adds a new score to the leaderboard.
    ///
    /// Scores are stored in insertion order internally. When displaying,
    /// they will be sorted by number of guesses.
    ///
    /// # Arguments
    /// * `score` - The Score to add
    ///
    /// # Examples
    /// ```
    /// use d7082e_lab2::{Leaderboard, Score};
    /// let mut leaderboard = Leaderboard::new();
    /// leaderboard.add_score(Score::new("Alice".to_string(), 5));
    /// assert_eq!(leaderboard.len(), 1);
    /// ```
    pub fn add_score(&mut self, score: Score) {
        self.scores.push(score);
    }

    /// Returns the number of scores in the leaderboard.
    pub fn len(&self) -> usize {
        self.scores.len()
    }

    /// Checks if the leaderboard is empty.
    pub fn is_empty(&self) -> bool {
        self.scores.is_empty()
    }

    /// Returns a sorted copy of the scores.
    ///
    /// Scores are sorted first by number of guesses (ascending), then by
    /// insertion order (most recent first for ties).
    ///
    /// # Examples
    /// ```
    /// use d7082e_lab2::{Leaderboard, Score};
    /// let mut leaderboard = Leaderboard::new();
    /// leaderboard.add_score(Score::new("Alice".to_string(), 5));
    /// leaderboard.add_score(Score::new("Bob".to_string(), 5));
    /// leaderboard.add_score(Score::new("Charlie".to_string(), 3));
    ///
    /// let sorted = leaderboard.get_sorted();
    /// assert_eq!(sorted[0].guesses, 3);  // Charlie with 3 guesses first
    /// assert_eq!(sorted[1].name, "Bob");  // Bob before Alice (later insertion)
    /// assert_eq!(sorted[2].name, "Alice");
    /// ```
    pub fn get_sorted(&self) -> Vec<Score> {
        let mut sorted = self.scores.clone();

        // Sort by guesses ascending, then by insertion order (reverse index = most recent first)
        sorted.sort_by(|a, b| {
            match a.guesses.cmp(&b.guesses) {
                Ordering::Equal => {
                    // For ties, maintain reverse insertion order (most recent first)
                    // We find the indices in the original list
                    let a_index = self.scores.iter().position(|s| s == a);
                    let b_index = self.scores.iter().position(|s| s == b);

                    match (a_index, b_index) {
                        (Some(ai), Some(bi)) => bi.cmp(&ai), // Reverse order
                        _ => Ordering::Equal,
                    }
                }
                other => other,
            }
        });

        sorted
    }

    /// Loads a leaderboard from the default high scores file.
    ///
    /// Returns an empty leaderboard if the file doesn't exist.
    ///
    /// # Errors
    /// Returns an error if the file exists but cannot be read or parsed.
    ///
    /// # Examples
    /// ```no_run
    /// use d7082e_lab2::Leaderboard;
    /// let leaderboard = Leaderboard::load().expect("Failed to load");
    /// ```
    pub fn load() -> io::Result<Self> {
        if !Path::new(SCORE_FILE).exists() {
            return Ok(Leaderboard::new());
        }

        let contents = fs::read_to_string(SCORE_FILE)?;
        let leaderboard = serde_json::from_str(&contents)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        Ok(leaderboard)
    }

    /// Saves the leaderboard to the default high scores file.
    ///
    /// # Errors
    /// Returns an error if the file cannot be written.
    ///
    /// # Examples
    /// ```no_run
    /// use d7082e_lab2::{Leaderboard, Score};
    /// let mut leaderboard = Leaderboard::new();
    /// leaderboard.add_score(Score::new("Alice".to_string(), 5));
    /// leaderboard.save().expect("Failed to save");
    /// ```
    pub fn save(&self) -> io::Result<()> {
        let json = serde_json::to_string_pretty(&self)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        fs::write(SCORE_FILE, json)?;
        Ok(())
    }

    /// Returns a formatted string representation of the leaderboard.
    ///
    /// Displays scores sorted from lowest to highest, with right-aligned
    /// score numbers and left-aligned names.
    ///
    /// # Examples
    /// ```
    /// use d7082e_lab2::{Leaderboard, Score};
    /// let mut leaderboard = Leaderboard::new();
    /// leaderboard.add_score(Score::new("Alice".to_string(), 5));
    /// leaderboard.add_score(Score::new("Bob".to_string(), 3));
    /// let display = leaderboard.format_display();
    /// assert!(display.contains("Bob"));
    /// assert!(display.contains("Alice"));
    /// ```
    pub fn format_display(&self) -> String {
        let sorted = self.get_sorted();

        let mut result = String::from("\n  All time best scores!\n");
        result.push_str("   #   Name\n");
        result.push_str("  --   --------\n");

        for (_idx, score) in sorted.iter().enumerate() {
            result.push_str(&format!("  {:2}   {}\n", score.guesses, score.name));
        }

        result
    }
}

impl Default for Leaderboard {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_creation() {
        let score = Score::new("Alice".to_string(), 5);
        assert_eq!(score.name, "Alice");
        assert_eq!(score.guesses, 5);
    }

    #[test]
    fn test_leaderboard_add_score() {
        let mut leaderboard = Leaderboard::new();
        leaderboard.add_score(Score::new("Alice".to_string(), 5));
        assert_eq!(leaderboard.len(), 1);
    }

    #[test]
    fn test_leaderboard_empty() {
        let leaderboard = Leaderboard::new();
        assert!(leaderboard.is_empty());
        assert_eq!(leaderboard.len(), 0);
    }

    #[test]
    fn test_sorting_by_guesses() {
        let mut leaderboard = Leaderboard::new();
        leaderboard.add_score(Score::new("Anki".to_string(), 1));
        leaderboard.add_score(Score::new("Kalle".to_string(), 3));
        leaderboard.add_score(Score::new("Olle".to_string(), 4));
        leaderboard.add_score(Score::new("Arne".to_string(), 13));

        let sorted = leaderboard.get_sorted();
        assert_eq!(sorted[0].guesses, 1);
        assert_eq!(sorted[1].guesses, 3);
        assert_eq!(sorted[2].guesses, 4);
        assert_eq!(sorted[3].guesses, 13);
    }

    #[test]
    fn test_sorting_with_ties_recent_first() {
        // When scores are equal, most recent (last added) should appear first
        let mut leaderboard = Leaderboard::new();
        leaderboard.add_score(Score::new("Kalle".to_string(), 4));
        leaderboard.add_score(Score::new("Kalle".to_string(), 4));
        leaderboard.add_score(Score::new("Olle".to_string(), 4));

        let sorted = leaderboard.get_sorted();

        // All have 4 guesses, but Olle (most recent) should be first, then the Kalles in reverse order
        assert_eq!(sorted[0].name, "Olle");
        assert_eq!(sorted[1].name, "Kalle");
        assert_eq!(sorted[2].name, "Kalle");
    }

    #[test]
    fn test_complex_sorting_scenario() {
        // Test the exact example from the README
        let mut leaderboard = Leaderboard::new();
        leaderboard.add_score(Score::new("Anki".to_string(), 1));
        leaderboard.add_score(Score::new("Kalle".to_string(), 3));
        leaderboard.add_score(Score::new("Kalle".to_string(), 4));
        leaderboard.add_score(Score::new("Kalle".to_string(), 4));
        leaderboard.add_score(Score::new("Olle".to_string(), 4));
        leaderboard.add_score(Score::new("Arne".to_string(), 13));

        let sorted = leaderboard.get_sorted();

        assert_eq!(sorted[0].guesses, 1);
        assert_eq!(sorted[0].name, "Anki");

        assert_eq!(sorted[1].guesses, 3);
        assert_eq!(sorted[1].name, "Kalle");

        // For the three 4-guess entries: Olle was added last, so appears first
        assert_eq!(sorted[2].guesses, 4);
        assert_eq!(sorted[2].name, "Olle");

        // Then the two Kalles in reverse insertion order
        assert_eq!(sorted[3].guesses, 4);
        assert_eq!(sorted[3].name, "Kalle");

        assert_eq!(sorted[4].guesses, 4);
        assert_eq!(sorted[4].name, "Kalle");

        assert_eq!(sorted[5].guesses, 13);
        assert_eq!(sorted[5].name, "Arne");
    }

    #[test]
    fn test_leaderboard_serialization() {
        let mut leaderboard = Leaderboard::new();
        leaderboard.add_score(Score::new("Alice".to_string(), 5));
        leaderboard.add_score(Score::new("Bob".to_string(), 3));

        let json = serde_json::to_string(&leaderboard).unwrap();
        let deserialized: Leaderboard = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.len(), 2);
        let sorted = deserialized.get_sorted();
        assert_eq!(sorted[0].guesses, 3);
        assert_eq!(sorted[1].guesses, 5);
    }

    #[test]
    fn test_format_display() {
        let mut leaderboard = Leaderboard::new();
        leaderboard.add_score(Score::new("Alice".to_string(), 5));
        leaderboard.add_score(Score::new("Bob".to_string(), 3));

        let display = leaderboard.format_display();
        assert!(display.contains("Alice"));
        assert!(display.contains("Bob"));
        assert!(display.contains("All time best scores"));
        assert!(display.contains("3"));
        assert!(display.contains("5"));
    }
}
