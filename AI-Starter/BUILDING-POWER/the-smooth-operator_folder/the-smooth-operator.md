the-smooth-operator
available
This exercise is not in the time scope anymore so it won't reward you any XP.
Level 5
required
XP
Files to submit
Allowed functions
0.00 B
the-smooth-operator.js
—
Mindful AI mode

AI-Powered Learning Techniques
Step-by-Step Instruction Technique:

This type of prompt asks AI to break down complex concepts into smaller, manageable steps. When you encounter a challenging concept, use this technique to get a clear learning path that builds understanding gradually.

Find examples across the subject and use AI to break down confusing parts into steps! ;)

Math Operators
In JavaScript, operators are symbols that perform operations on variables and values. Let's explore the most common types of operators you'll encounter.

For now, let's focus on the ones you probably already know:

+ Addition

- Subtraction

/ Division

* Multiplication

** Exponentiation (power)

These operators are used in the same way we write them in math:

console.log(5 + 7); // -> 12
console.log(5 * 5); // -> 25
console.log(7 - 5); // -> 2
console.log(9 / 3); // -> 3
console.log(2 ** 3); // -> 8 (2 to the power of 3)
Operators are evaluated using classic precedence rules:

console.log(1 + 5 * 10); // -> 51 (multiplication happens first)
You can use parentheses () to enforce precedence:

console.log((1 + 5) * 10); // -> 60 (addition happens first)
Operators produce a value, so they can be assigned to variables:

let halfMyAge = 33 / 2;
let twiceMyAge = 33 * 2;
let ageSquared = 33 ** 2;
Prompt example:
"Break down operator precedence in JavaScript step-by-step. Show me examples of expressions with multiple operators and walk me through how JavaScript evaluates each step."

Template Literals
JavaScript allows you to include expressions within strings using template literals. This is done using backticks ` and the ${} syntax to include variable values or calculations.

Example
let name = "Robot";
let age = 5;
console.log(`Hello, my name is ${name} and I'm ${age} years old.`);
// Output: Hello, my name is Robot and I'm 5 years old.
You can also include calculations inside the ${}:

console.log(`5 + 10 = ${5 + 10}`); // -> 5 + 10 = 15
Note: This only works with backticks `, not with double quotes " or single quotes '.

Prompt example:
"Show me step-by-step how to build template literals in JavaScript. Start with simple variable insertion, then show me how to include calculations and more complex expressions."

Instructions
Task 1:
A variable called smooth has been provided for you. Use math operators to create the following variables:

lessSmooth: This is 1 less than smooth

semiSmooth: This is half the value of smooth

plus11: This is smooth plus 11

ultraSmooth: This is smooth squared

Expected format:

let lessSmooth = // your calculation here
let semiSmooth = // your calculation here
let plus11 = // your calculation here
let ultraSmooth = // your calculation here
Task 2:
Two variables name and age have been provided for you. Use template literals to create:

presentation: A string that outputs exactly: Hello, my name is [NAME] and I'm [AGE] years old. where [NAME] and [AGE] are replaced with the actual variable values.

Expected format:

let presentation = // your template literal here using backticks and ${}
BGM:
Sade - Smooth Operator - Official - 1984