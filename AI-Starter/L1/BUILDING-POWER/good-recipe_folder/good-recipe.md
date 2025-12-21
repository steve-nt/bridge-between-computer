good-recipe
available
This exercise is not in the time scope anymore so it won't reward you any XP.
Level 6
required
XP
Files to submit
Allowed functions
0.00 B
good-recipe.js
—
Mindful AI Mode

Context
Welcome to this quest! I bet you're hungry after a long journey gaining power!

Are you ready to cook? Instead of cooking food, we'll be applying recipes to data using methods! In JavaScript, methods are special functions that you call on values to transform or analyze them.

Let's dive in!

AI-Powered Learning Techniques
Guided Practice Technique: This type of prompt asks AI to walk you through exercises step-by-step, providing practice problems with immediate explanations. Use this technique to build confidence by practicing concepts in small, manageable steps.

Find examples throughout the subject. ;)

Concepts
Methods
Methods are functions called on values using dot notation. They transform or analyze data:

let num = 10 / 3;
console.log(num.toFixed(2)); // -> '3.33' (formats number)

let text = "hello";
console.log(text.toUpperCase()); // -> 'HELLO' (converts to uppercase)
String Methods
Slice - Extract parts of a string:

let alphabet = "abcdefghij";
console.log(alphabet.slice(3)); // -> 'defghij' (from index 3 to end)
console.log(alphabet.slice(0, -2)); // -> 'abcdefgh' (remove last 2)
console.log(alphabet.slice(2, -1)); // -> 'cdefghi' (remove first 2 and last 1)
Case conversion:

let message = "Hello World";
console.log(message.toLowerCase()); // -> 'hello world'
console.log(message.toUpperCase()); // -> 'HELLO WORLD'
Math.max Function
Finds the largest number from multiple arguments:

console.log(Math.max(5, 10, 3)); // -> 10
console.log(Math.max(robot1.age, robot2.age)); // -> largest age
Prompt example:
"Walk me through using string methods step-by-step. Give me a practice exercise with .slice(), show me the solution, then let me try a similar problem with .toUpperCase() and provide feedback on my approach."

Instructions
Task 1:
Three robot objects martin, kevin, and stephanie have been provided, each with an age property.

Declare an oldestAge variable that uses Math.max() to find the highest age value among the three robots:

let oldestAge = // use Math.max with martin.age, kevin.age, stephanie.age
Task 2:
An alphabet variable (containing the alphabet string) has been provided. Use the .slice() method to create three new variables:

cutFirst: Removing the first 10 characters (start slicing from index 10).

cutLast: Removing the last 3 characters.

cutFirstLast: Removing first 5 and last 6 characters.

Task 3:
A message variable has been provided. Use string case methods to create two variables:

noCaps: Converting message to lowercase using .toLowerCase()

allCaps: Converting message to uppercase using .toUpperCase()