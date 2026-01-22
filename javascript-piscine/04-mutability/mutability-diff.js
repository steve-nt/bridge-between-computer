const person = {
    name: 'Rick',
    age: 78,
    country: 'FR',
}

const clone1 = { ...person } // Spread operator - SHALLOW COPY
const clone2 = Object.assign({}, person) // Object.assign - SHALLOW COPY  
const samePerson = person // REFERENCE - SAME OBJECT

console.log('BEFORE mutation:');
console.log('Original:', person);
console.log('Clone1:', clone1);
console.log('Clone2:', clone2); 
console.log('Same:', samePerson);
console.log('---');

// MUTATE original
person.age = 79;
person.country = 'USA';

console.log('AFTER person.age=79, person.country="USA":');
console.log('Original:', person);      // Changed: {name: 'Rick', age: 79, country: 'USA'}
console.log('Clone1:', clone1);        // UNCHANGED: {name: 'Rick', age: 78, country: 'FR'}
console.log('Clone2:', clone2);        // UNCHANGED: {name: 'Rick', age: 78, country: 'FR'}
console.log('Same:', samePerson);      // Changed: {name: 'Rick', age: 79, country: 'USA'}
