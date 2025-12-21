colorful-arms
available
This exercise is not in the time scope anymore so it won't reward you any XP.
Level 7
required
XP
Files to submit
Allowed functions
0.00 B
colorful-arms.html, colorful-arms.css, colorful-arms.js
—
JS-Powered Mode

Context
Your robot is already impressive, but you can make it even more eye-catching! At the very least, you can make it show off its colorful arms to the world!

Follow the instructions, ask your peers if you get stuck, and stay motivated because you're close to the goal! Pay attention to every hint provided in the subject! Continue working on the code from the last exercise, and update the file names accordingly!

AI-Powered Learning Techniques
Integration Technique: This approach uses AI to help you understand how to combine multiple interactive features into one cohesive application. When adding new functionality to existing code, ask AI to help you see how different event handlers work together.

Instructions
By completing these tasks, you will enable your robot's arms to change to beautiful colors!

Task 1:
Add another button to the top-right corner of the page with the ID arm-color. This button will change the color of the robot's arms. Add the following to the HTML structure:

<button id="arm-color">Change robot's arm colors</button>
Style the button in the CSS file:

#arm-color {
  position: fixed;
  top: 170px;
  right: 30px;
  padding: 10px;
  z-index: 2;
}
Replace the existing button element styles with this block:

button {
  border-radius: 50px;
  padding: 10px;
  z-index: 1;
  position: fixed;
  top: 30px;
  right: 30px;
}
Notice how much prettier the buttons look now!

Task 2:
In the JavaScript file, as in the previous exercise, select the following elements:

The button with the ID arm-color.

The left and right arm elements with the IDs arm-left and arm-right.

Task 3:
In the code example below, we demonstrate the variable randomColor, which generates a random color each time it is used.

Create a function that applies the same random color to both arms by changing their backgroundColor whenever the function is called.

Add an event listener for the click event on the button with the ID arm-color.

Code Example:
// Select the button with id 'arm-color'
//...Here

// Select the left and right arm elements
//...Here

// Your function that gets triggered when clicking the new button
const handleChangeArmColor = (event) => {
  // Generate a random color
  const randomColor = `#${Math.floor(Math.random() * 16777215).toString(16)}`;

  // Apply the random color to both arms
  //...Here
};

// Attach the handleChangeArmColor function to the 'arm-color' button
armColorButton.addEventListener("click", handleChangeArmColor);
Expected Result
You can see an example of the expected result here.

Prompt example:

"I'm building a web application with multiple interactive buttons that each control different parts of a robot. Can you help me understand how to organize multiple event listeners and ensure they don't interfere with each other?"