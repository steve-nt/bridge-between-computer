// Function to change arm colors
const changeArmColor = (robotClass) => {
  const randomColor = `#${Math.floor(Math.random() * 16777215).toString(16).padStart(6, '0')}`;
  document.querySelector(`.${robotClass} #arm-left`).style.backgroundColor = randomColor;
  document.querySelector(`.${robotClass} #arm-right`).style.backgroundColor = randomColor;
};

// Function to change leg colors
const changeLegColor = (robotClass) => {
  const randomColor = `#${Math.floor(Math.random() * 16777215).toString(16).padStart(6, '0')}`;
  document.querySelector(`.${robotClass} #leg-left`).style.backgroundColor = randomColor;
  document.querySelector(`.${robotClass} #leg-right`).style.backgroundColor = randomColor;
};

// Function to change eye colors
const changeEyeColor = (robotClass) => {
  const randomColor = `#${Math.floor(Math.random() * 16777215).toString(16).padStart(6, '0')}`;
  document.querySelector(`.${robotClass} #eye-left`).style.backgroundColor = randomColor;
  document.querySelector(`.${robotClass} #eye-right`).style.backgroundColor = randomColor;
};

// Function to change face colors
const changeFaceColor = (robotClass) => {
  const randomColor = `#${Math.floor(Math.random() * 16777215).toString(16).padStart(6, '0')}`;
  document.querySelector(`.${robotClass} #eyes`).style.backgroundColor = randomColor;
};

// Event listener for keyboard input
document.addEventListener("keydown", function (event) {
  // Steven's robot controls (1, 2, Q, A)
  if (event.key === "1") {
    changeArmColor("steven-robot");
  }
  if (event.key === "2") {
    changeLegColor("steven-robot");
  }
  if (event.key === "Q" || event.key === "q") {
    changeEyeColor("steven-robot");
  }
  if (event.key === "A" || event.key === "a") {
    changeFaceColor("steven-robot");
  }

  // Alice's robot controls (3, 4, W, S)
  if (event.key === "3") {
    changeArmColor("alice-robot");
  }
  if (event.key === "4") {
    changeLegColor("alice-robot");
  }
  if (event.key === "W" || event.key === "w") {
    changeEyeColor("alice-robot");
  }
  if (event.key === "S" || event.key === "s") {
    changeFaceColor("alice-robot");
  }

  // Bob's robot controls (5, 6, E, D)
  if (event.key === "5") {
    changeArmColor("bob-robot");
  }
  if (event.key === "6") {
    changeLegColor("bob-robot");
  }
  if (event.key === "E" || event.key === "e") {
    changeEyeColor("bob-robot");
  }
  if (event.key === "D" || event.key === "d") {
    changeFaceColor("bob-robot");
  }
});
