# Genetic

Simple genetic algorithm library designed to make optimizing your problem easier.
Comes with some standard mutation, crossover / breeding functions, simple fitness function and tournament selection. Written in C++.
The library is just a couple of header files that you can include in your own project to start using.

Just a fun hobby project.

Check out the examples to see how you can use this.

Also checkout my other project "Kop" to see this in action for keyboard layout optimization.

## Examples
To run the examples, just use `make run`. That will compile and run the examples.
You should see the iterations and best fit at each iteration. and a final evaluation.

1. The easom_wrap example shows how you can create a custom data type to be used. The example just wraps around a standard double but you can take this further.
2. The xor example shows how you can train a neural network to calculate XOR.
3. the xor_ind examples shows how you can individually call the steps of the genetic algorithm instead of the letting the engine simulate it all. Useful when you want to add in your own hooks.


