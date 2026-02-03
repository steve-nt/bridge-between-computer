# Lab 2 Implementation Guide - Guessing Game Extension

## Overview
This document explains the implementation of a Rust-based number guessing game with persistent high score tracking, comprehensive error handling, and thorough documentation.

## What Was Implemented

### 1. **High Score System (lib.rs)**
A library module with two core structures for managing game scores.

#### Key Components:
- **`Score` Struct**: Represents a single high score entry with player name and guess count
  - Implements `Serialize` and `Deserialize` for JSON persistence
  - Implements `Debug`, `Clone`, and `PartialEq` for testing and comparison

- **`Leaderboard` Struct**: Manages a collection of scores
  - **`add_score()`**: Adds new scores in insertion order
  - **`get_sorted()`**: Returns scores sorted by guesses (ascending), with ties broken by insertion order (most recent first)
  - **`format_display()`**: Returns a formatted leaderboard string with right-aligned scores and left-aligned names
  - **`load()`**: Loads scores from `high_scores.json` (returns empty leaderboard if file missing)
  - **`save()`**: Persists leaderboard to JSON file with proper error handling

#### Sorting Algorithm Details:
The sorting logic ensures that:
1. Lower guess counts appear first (ascending order)
2. When two players have the same score, the most recently added (latest) score appears first
3. Implementation: Uses `sort_by()` with pattern matching to compare scores, falling back to reverse index comparison for ties

```rust
sorted.sort_by(|a, b| {
    match a.guesses.cmp(&b.guesses) {
        Ordering::Equal => {
            let a_index = self.scores.iter().position(|s| s == a);
            let b_index = self.scores.iter().position(|s| s == b);
            match (a_index, b_index) {
                (Some(ai), Some(bi)) => bi.cmp(&ai), // Reverse order (most recent first)
                _ => Ordering::Equal,
            }
        }
        other => other,
    }
});
```

### 2. **Main Game Loop (main.rs)**
Extended the basic guessing game to support multiple rounds with score tracking.

#### Key Functions:
- **`get_user_text_input()`**: Safely reads non-empty text input from the user
- **`get_user_guess()`**: Reads and validates numeric input, supporting quit command (0)
- **`play_round()`**: Executes a single game round with feedback
  - Returns the number of guesses on win, 0 on quit
  - Generates random number in range 1-10 per requirements
  - Compares using `cmp()` and provides feedback (too high/low/correct)

- **`main()`**: Orchestrates the complete game flow
  - Loads existing leaderboard on startup
  - Loops until user quits
  - Collects player name after win
  - Saves leaderboard after each win
  - Displays formatted leaderboard
  - Asks if player wants another round

#### Game Range: **1-10** (reduced from original 1-100)
Easier difficulty per requirements, allowing reasonable high score comparisons.

### 3. **File Persistence**

#### Storage Format: JSON
```json
{
  "scores": [
    { "name": "Alice", "guesses": 5 },
    { "name": "Bob", "guesses": 3 }
  ]
}
```

**Why JSON?**
- Human-readable and inspectable
- Built-in support via `serde_json` crate
- Type-safe deserialization
- Industry-standard format

#### File Handling:
- **Filename**: `high_scores.json` (in current working directory)
- **On Startup**: Loads existing file; creates empty leaderboard if missing
- **After Each Win**: Saves updated leaderboard immediately
- **Error Handling**: Non-critical save failures print warnings but don't crash

### 4. **Error Handling**

The implementation follows Rust's error handling principles:

#### No Panics in Normal Operation
- User input validation never panics
- File I/O wrapped in `io::Result` with proper error propagation
- JSON parsing wrapped in error conversion

#### Recoverable Errors (Result Types)
```rust
pub fn load() -> io::Result<Self> { ... }
pub fn save(&self) -> io::Result<() { ... }
```

#### Unrecoverable Errors (expect/unwrap)
Only used for operations that should never fail:
```rust
io::stdout().flush().expect("Failed to flush stdout");
```

**Justification**: stdout flushing is a system-level operation. If it fails, the environment is severely compromised and further execution is meaningless.

### 5. **Testing**

Comprehensive unit tests in `lib.rs` cover:
- Basic functionality (creation, adding scores, empty leaderboard)
- Sorting by guess count
- Tie-breaking by insertion order (most recent first)
- Complex scenario from README specification
- Serialization/deserialization round-trip
- Display formatting

**Key Test: `test_complex_sorting_scenario()`**
Verifies the exact example from requirements:
```
All time best scores!
 #   Name
--   --------
 1   Anki
 3   Kalle
 4   Olle      (added last, appears first among ties)
 4   Kalle
 4   Kalle
13   Arne
```

### 6. **Documentation**

#### Doc Comments (`///`)
- Every public function has documentation
- Includes purpose, arguments, return values, and examples
- Doc tests verify examples compile and run correctly

#### Module Documentation (`//!`)
- Explains purpose and responsibility of each module
- Provides context for users of the library

#### Code Comments
Minimal, only where logic is non-obvious:
- Sorting algorithm explanation
- Unrecoverable error justifications

## Dependencies

### External Crates
- **`rand`**: Random number generation (already in Lab 1)
- **`serde`**: Serialization framework
- **`serde_json`**: JSON support

**Why these?**
- Minimal dependencies (only 2 new)
- `serde` is the Rust standard for serialization
- Avoids reinventing serialization logic
- Both are well-maintained and widely used

## Rust Concepts Applied

### 1. **Ownership & Borrowing**
- `Leaderboard` owns the `Vec<Score>`
- Methods borrow `&self` for read operations, `&mut self` for modifications
- Return types carefully chosen to avoid unnecessary copies

### 2. **Pattern Matching**
- Match expressions for `Result` and `Ordering` types
- Destructuring in `get_sorted()` tie-breaking logic

### 3. **Error Handling**
- `Result<T, E>` for fallible operations
- Error conversion with `map_err()` for JSON parsing
- `unwrap_or_else()` for graceful degradation

### 4. **Traits**
- `Serialize` / `Deserialize` (serde)
- `Default` implementation for `Leaderboard`
- `Clone`, `Debug`, `PartialEq` for `Score`

### 5. **Generic Programming**
- `Vec<Score>` for type-safe collections
- Generic iteration with `iter()` and `enumerate()`

### 6. **Module System**
- `lib.rs` separates library concerns
- `main.rs` imports and uses library functionality
- Clean separation of concerns

## Key Design Decisions

### 1. **Leaderboard Stores Insertion Order**
Decision: Store scores in insertion order internally, sort on display
- Allows efficient addition (O(1) append)
- Enables correct tie-breaking behavior
- Display sorting is rare vs. adding scores

### 2. **Single JSON File**
Decision: One file per game instance (in current directory)
- Simplicity for learning project
- Matches typical usage (one game per directory)
- Alternative: Could use user-specific files or centralized directory

### 3. **No Interactive Play-Again Loop at OS Level**
Decision: Simple yes/no check, then continue or exit
- Keeps code readable
- Meets requirements
- Could be enhanced with menu system later

### 4. **File I/O Not Cached**
Decision: Load on startup, save after each win
- Ensures no data loss if game crashes
- Acceptable performance for this scale
- Alternative: Could use atomic writes for safety

## Testing Strategy

### Unit Tests (8 total in lib.rs)
Test library functionality in isolation:
- Edge cases (empty leaderboard, single entry)
- Sorting correctness (including tie-breaking)
- Serialization round-trips
- Display formatting

### Doc Tests (7 in lib.rs)
Verify code examples work:
- All public function examples compile and run
- Demonstrates API usage
- Double-checks documentation accuracy

### Manual Testing
Recommended interactive testing:
1. Play one round, verify score saved
2. Play multiple rounds, check sorting
3. Quit mid-game, verify file unchanged
4. Delete high_scores.json, start fresh
5. Verify display formatting with various names/scores

## Code Statistics

| Component | Lines | Purpose |
|-----------|-------|---------|
| lib.rs | ~320 | Score management, leaderboard, persistence |
| main.rs | ~160 | Game loop, user interaction |
| Tests | ~180 | Unit and doc tests |
| **Total** | ~660 | Complete implementation |

## Compilation & Execution

```bash
# Build (debug)
cargo build

# Run with optimizations
cargo build --release

# Run tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run the game
cargo run
```

## Learning Outcomes

### Rust Mastery
- **Ownership**: Understood when to move, borrow, or clone
- **Error Handling**: Appreciated `Result` over exceptions
- **Collections**: Worked with `Vec` and sorting
- **Trait System**: Used serde traits effectively

### Software Design
- **Separation of Concerns**: Library vs. binary
- **API Design**: Clear function signatures, good documentation
- **Testing**: Comprehensive coverage of edge cases
- **Error Handling**: Graceful degradation where appropriate

### Tooling
- **Cargo**: Managing dependencies, building, testing
- **External Crates**: Integrating serde ecosystem
- **Documentation**: Writing and testing doc comments

## Keywords & Search Terms

For further learning:
- **Rust Sorting**: `Vec::sort_by()`, `Ordering` trait
- **Serialization**: `serde`, `serde_json`, derive macros
- **Error Handling**: `Result<T, E>`, `map_err()`, error conversion
- **File I/O**: `std::fs`, `read_to_string()`, `write()`
- **Traits**: `Serialize`, `Deserialize`, `Default`
- **Pattern Matching**: `match` expressions, destructuring
- **Testing**: `#[test]`, `#[cfg(test)]`, doc tests

## References

### Official Resources
- [The Rust Book](https://doc.rust-lang.org/book/) - Chapters 7-12 (modules, error handling)
- [Rust Standard Library](https://doc.rust-lang.org/std/) - Collections, I/O, iterators
- [serde Documentation](https://serde.rs/) - Serialization framework
- [serde_json Guide](https://docs.rs/serde_json/) - JSON support

### Key Chapters Referenced
- Chapter 7: Managing Growing Projects (modules)
- Chapter 8: Collections (Vec, iterators)
- Chapter 9: Error Handling (Result, custom errors)
- Chapter 10: Generic Types (trait bounds)
- Chapter 11: Testing (unit tests, doc tests)
- Chapter 12: An I/O Project (file handling)

## Potential Enhancements

### Future Features
1. **Global Leaderboard**: Share scores across users
2. **Difficulty Levels**: Different ranges for different modes
3. **Timed Rounds**: Challenge mode with time limits
4. **Statistics**: Win percentage, average guesses, streak
5. **GUI**: Move from CLI to graphical interface
6. **Database**: Replace JSON with SQLite for scalability

### Code Improvements
1. **Configuration File**: Customize game range, file location
2. **Custom Error Types**: Better error context and recovery
3. **Async I/O**: Non-blocking file operations
4. **Logging**: Debug/info output control via logging framework
5. **Benchmarking**: Performance testing of sorting

## Conclusion

This Lab 2 implementation demonstrates:
- ✅ Rust's safety and error handling capabilities
- ✅ Modular design with clear separation of concerns
- ✅ Comprehensive testing for correctness
- ✅ Professional documentation practices
- ✅ Persistent data management
- ✅ Robust, crash-resistant error handling

The guessing game evolved from a simple one-round game to a fully-featured application with persistent high score tracking, comprehensive documentation, and zero-panic error handling.
