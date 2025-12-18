play-with-variables
available
This exercise is not in the time scope anymore so it won't reward you any XP.
Level 4
required
XP
Files to submit
Allowed functions
0.00 B
play-with-variables.js
—
Mindful AI mode

Context
Remember that if things get a little hectic at times, take the time to get closer to your peers so that you can think, share and move forward together.

Keep Going!

AI-Powered Learning Techniques
Clarification Technique:

This type of prompt asks AI to break down confusing concepts into simpler parts and provide multiple ways to understand the same idea. When you encounter something that seems unclear, use this technique to get different explanations and examples until the concept clicks.

Find examples across the subject and use AI to clarify anything that feels confusing! ;)

Escape characters
Quote delimiters can be tricky to work with when you need to include quotes inside your text.

Since quotes are used to mark the beginning and end of strings, JavaScript needs a special way to include actual quote characters inside the text itself.

For example, if you want to include a single quote in your text:

console.log('I can't do this!'); // This breaks because of the apostrophe
The backslash \ is used to escape special characters. When you put \ before a character, JavaScript treats it as a literal character instead of a special symbol:

console.log("I can't do this!"); // This works!
// Output: I can't do this!
You can escape any quote type:

console.log('She said "Hello!"'); // Escaping double quotes
console.log(`The file is in \`code\` format`); // Escaping backticks
console.log("It's a beautiful day"); // Escaping single quotes
Prompt example:
"I'm confused about escape characters in JavaScript strings. Can you explain it in 3 different ways: using a simple analogy, showing me step-by-step what happens, and giving me practical examples of when I'd need this?"

Assign and re-assign
The let keyword is used to declare new variables for the first time.

Once a variable is declared, you cannot declare it again with the same name:

let robotName = "R2D2";
let robotName = "C3PO"; // Error! robotName already exists
However, you can re-assign a new value to an existing variable using just the assignment operator =:

let robotName = "R2D2";
robotName = "C3PO"; // This works! No 'let' needed for reassignment
console.log(robotName); // Output: C3PO
Important differences:

const creates variables that cannot be re-assigned (constant values)

var is old syntax from before 2015 - avoid using it as it has problematic behavior

const pi = 3.14159;
pi = 3.14; // Error! Cannot reassign const variables

let temperature = 20;
temperature = 25; // This works fine with let
Prompt example:
"I'm getting confused about when to use let vs when to just use = for variables in JavaScript. Can you clarify this by showing me examples of declaring, reassigning, and what happens when I make mistakes?"

Instructions
Task 1:
Create two variables to practice escape characters:

Create an escapeFromDelimiters variable. When logged to the console, it should output: Here are the quotes:, followed by , ", and ' (comma, double quote, and single quote), exactly as shown, without the parentheses.

Create an escapeTheEscape variable. When logged to console, it should output exactly: Here is a backslash: \

Task 2:
Practice variable reassignment:

A variable called power has already been declared for you.

Reassign the power variable to the string value 'levelMax' (remember: no let keyword needed for reassignment).

"How did I escape? With difficulty. How did I plan this moment? With pleasure." \

Alexandre Dumas, The Count of Monte Cristo