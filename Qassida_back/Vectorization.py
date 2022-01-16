from numpy import argmax

cleaned = open('CleanPoems.txt', 'r+',encoding="utf-8")


text = cleaned.read()
print(text[:100])
vocab = set(text)
print(len(vocab), vocab)
print()
char_to_int = dict((l, e) for e, l in enumerate(vocab))
int_to_char = dict((e, l) for e, l in enumerate(vocab))

integer_encoded = [char_to_int[char] for char in text[:10]]
print( " ' ' is encoded as :", char_to_int[" "])
print(integer_encoded)

onehot_encoded = list()
for value in integer_encoded:
	letter = [0 for _ in range(len(vocab))]
	letter[value] = 1
	onehot_encoded.append(letter)
print(onehot_encoded)
# invert encoding
inverted = int_to_char[argmax(onehot_encoded[0])]
print(inverted)


cleaned.close()