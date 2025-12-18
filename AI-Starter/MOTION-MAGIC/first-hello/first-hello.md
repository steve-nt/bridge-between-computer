first-hello
available
This exercise is not in the time scope anymore so it won't reward you any XP.
Level 7
required
XP
Files to submit
Allowed functions
0.00 B
first-hello.html, first-hello.css, first-hello.js
—
JS-Powered Mode

Context
Your robot winked, but you can make it talk too! Or at least, make it say its first hello to the world!

Follow the instructions, ask your peers if you are stuck, and stay motivated because you are close to your goal! Follow every hint you have in the subject! Continue on the code from the last exercise, and change the file names!

AI-Powered Learning Techniques
Problem-Building Technique: This approach uses AI to help you understand how to combine multiple concepts you've learned into more complex functionality. When building interactive features, ask AI to help you break down the problem and see how different DOM methods work together.

Instructions
Now that you know how to make your creation move, what about making it communicate its first words to the world?

After finishing these tasks, you will be able to display and hide Hello World in the torso of your robot by clicking on a second button.

Task 1:
Let's put a second button in the top right corner of the page that will add some text when clicked. Add it in the HTML structure:

<button id="speak-button">Click me to speak</button>
Add the button style in the CSS file:

button#speak-button {
  top: 100px;
}
Also add this class to style the text we will add:

.words {
  text-align: center;
  font-family: sans-serif;
}
Task 2:
In the JS file, like in the previous exercise, get the HTML button element with id speak-button and add an event listener on click event, triggering a function that will:

Select the torso element with id="torso".

Check if a div with the class words already exists inside the torso element.

If it exists, remove the existing div using removeChild.

Otherwise:

Create a new HTML element of type div using createElement.

Set its text content to "Hello World".

Add the 'words' class using classList.add().

Use the append method to add the new div inside the torso element.

Code Example:
// Select the button with id 'speak-button'

//...Here

//Your function that gets triggered when clicking the new button
const handleSpeakClick = (event) => {
  // Select the torso element where the text will be added or removed
  const body = document.querySelector("#torso");

  // Check if a div with the class 'words' already exists inside the torso
  const existingDiv = document.querySelector("#torso .words");

  if (existingDiv) {
    // If the "Hello World" text exists, remove it from the torso
  } else {
    // If the "Hello World" text does not exist, create and append it
    // Create a new div element
    // Add the 'words' class to the div
    // Set the text content to "Hello World"
    // Append the new div to the torso element
  }
};

// Attach the handleSpeakClick function to the 'speak-button' button
//...Here
Expected result
You can see an example of the expected result here

Prompt example:

"I'm building a feature that toggles text in and out of a div when a button is clicked. Can you help me understand how to combine createElement, classList.add, textContent, append, and removeChild to create this toggle functionality step by step?"
