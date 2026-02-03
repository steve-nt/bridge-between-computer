const person = {
    name: 'Rick',
    age: 78,
    country: 'FR',
}

const clone1 = { ...person } // Spread operator

const clone2 = Object.assign({}, person) // Object.assign

const samePerson = person // Reference assignment

console.log('Original person:', person)
console.log('Clone 1 (spread):', clone1)
console.log('Clone 2 (Object.assign):', clone2)
console.log('Same person (reference):', samePerson)

