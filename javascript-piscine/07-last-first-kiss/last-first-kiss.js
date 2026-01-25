function first(collection) {
    return collection[0];
}

function last(collection) {
    return collection[collection.length-1];
}

function kiss(collection) {
    return [last(collection), first(collection)];
}