const escapeStr = "`\\/\"'";

const arr = [4, '2'];

const obj = {
  str: "I am a string",
  num: 42,
  bool: true,
  undef: undefined
};

const nested = {
  arr: [4, undefined, '2'],
  obj: {
    str: "I am a string",
    num: 42,
    bool: true
  }
};

Object.freeze(arr);
Object.freeze(obj);
Object.freeze(nested);
Object.freeze(nested.obj);  // Freeze inner object too!