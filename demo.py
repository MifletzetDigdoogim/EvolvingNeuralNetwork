from models import Sequential, compute_loss
from layers import Layer


min_loss = 100
best_model = None

EPOCHS = 12
from random import uniform

while min_loss > 6.25:
    model = Sequential()
    first_layer = Layer(4, "sigmoid")
    model.add(first_layer)
    second_layer = Layer(5, "sigmoid")
    model.add(second_layer)
    third_layer = Layer(4, "softmax")
    model.add(third_layer)
    model.compile()

    loss = 0

    for i in range(EPOCHS):
        inpt = (uniform(-1, 1), uniform(-1, 1), uniform(-1, 1), uniform(-1, 1))
        expected_output = [1 if n == max(inpt) else 0 for n in inpt]
        output = model.run(inpt)
        loss += compute_loss(output, expected_output)

    if loss < min_loss:
        best_model = model
        min_loss = loss
        print("Loss is: " + str(loss))

# inp = (0.2, 0.2, 0.1, 0.3)
# r = best_model.run(inp)
# out = r.index(max(r)) + 1
# expected_output = [1 if n == max(inp) else 0 for n in inp]
# print("TEST: " + str(out) + " -- Expected: " + str(expected_output))
# inp = (0.4, 0.2, -0.1, 0.3)
# r = best_model.run(inp)
# out1 = r.index(max(r)) + 1
# expected_output = [1 if n == max(inp) else 0 for n in inp]
# print("TEST: " + str(out1) + " -- Expected: " + str(expected_output))
# inp = (-0.2, -0.2, -0.1, -0.3)
# r = best_model.run(inp)
# out2 = r.index(max(r)) + 1
# expected_output = [1 if n == max(inp) else 0 for n in inp]
# print("TEST: " + str(out2) + " -- Expected: " + str(expected_output))


while True:
    inp = []
    for i in range(4):
        g = input("Enter a number\n")
        inp.append(float(g))
    r = best_model.run(inp)
    out = r.index(max(r)) + 1
    print("Result is " + str(out))

