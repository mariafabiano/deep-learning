import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
plt.style.use('seaborn')


def word_argmax():
	# word vocab, argmax sampling
	with open('word_argmax_loss', 'r') as f:
		data = [line.split() for line in f.readlines()[12:-3]]

	print(len(data))
	data = [float(line[-1]) for line in data]

	# print(data)
	plt.plot(np.arange(len(data)), data)
	plt.title("Word-RNN with Argmax", fontsize=20, fontweight='bold')
	plt.xlabel("Epoch", fontweight='bold')
	plt.ylabel("Loss", fontweight='bold')
	plt.xticks(np.arange(0,  len(data)+1, 10), np.arange(0,  2*(len(data)+1), 20))
	plt.yticks(np.arange(0, round(max(data))+1, (round(max(data))+1)/10))
	plt.savefig("word_argmax_loss.png")
	plt.clf()


def word_sample():
	# word vocab, random sampling
	with open('word_sample_slurm.out', 'r') as f:
		data = [line.split() for line in f.readlines()[12:-6]]

	print(len(data))
	data = [float(line[-1]) for line in data]

	# print(data)
	plt.plot(np.arange(len(data)), data)
	plt.title("Word-RNN with Random Sampling", fontsize=20, fontweight='bold')
	plt.xlabel("Epoch", fontweight='bold')
	plt.ylabel("Loss", fontweight='bold')
	plt.xticks(np.arange(0,  len(data)+1, 10), np.arange(0,  2*(len(data)+1), 20))
	plt.yticks(np.arange(0, round(max(data))+1, (round(max(data))+1)/10))
	plt.savefig("word_sample_loss.png")
	plt.clf()


def char_argmax():
	# char vocab, argmax sampling
	plt.rcParams['figure.dpi'] = 300
	with open("char_losses.txt", 'r') as f:
		data = f.read().split(', ')

	data[0] = data[0][1:]
	data[-1] = data[-1][:-1]
	data = [float(loss) for loss in data]
	print(len(data))
	# print(data)

	plt.plot(np.arange(len(data)), data)
	plt.title("Char-RNN with Argmax", fontsize=20, fontweight='bold')
	plt.xlabel("Epoch", fontweight='bold')
	plt.ylabel("Loss", fontweight='bold')
	# plt.xticks(np.arange(0, len(data)+1, 10))
	plt.yticks(np.arange(0, round(max(data))+1, (round(max(data))+1)/10))
	plt.savefig("char_argmax_loss.png")
	plt.clf()


def word_temperature():
	# word vocab, temperature sampling
	with open('low_temperature_loss.log', 'r') as f:
		data = [float(line.split()[-1]) for line in f.readlines()[1:]]

	# print()
	# print(data)
	print(len(data))

	plt.plot(np.arange(len(data)), data)
	plt.title("Word-RNN with Temperature Sampling", fontsize=20, fontweight='bold')
	plt.xlabel("Epoch", fontweight='bold')
	plt.ylabel("Loss", fontweight='bold')
	step = 10
	xticks = np.arange(0, len(data)+1, step)
	plt.xticks(xticks, xticks*2)
	plt.yticks(np.arange(0, round(max(data))+1, (round(max(data))+1)/10))
	plt.savefig("word_temperature_loss.png")
	plt.clf()


def gru():
	with open('gru_loss.log', 'r') as f:
		data = [float(line.split()[-1]) for line in f.readlines()[1:]]

	# print()
	# print(data)
	print(len(data))

	plt.plot(np.arange(len(data)), data)
	plt.title("Word-RNN with GRU Cells", fontsize=20, fontweight='bold')
	plt.xlabel("Epoch", fontweight='bold')
	plt.ylabel("Loss", fontweight='bold')
	# step = 10
	# xticks = np.arange(0, len(data)+1, step)
	# print(np.arange(0,  400, len(xticks)//step))
	# plt.xticks(xticks, xticks*2)
	plt.yticks(np.arange(0, round(max(data))+1, (round(max(data))+1)/10))
	plt.savefig("gru_loss.png")
	plt.clf()


def four_layer():
	with open('four_layer_loss.log', 'r') as f:
		data = [float(line.split()[-1]) for line in f.readlines()[1:]]

	print(len(data))

	plt.plot(np.arange(len(data)), data)
	plt.title("Word-RNN with Four LSTM Cells", fontsize=20, fontweight='bold')
	plt.xlabel("Epoch", fontweight='bold')
	plt.ylabel("Loss", fontweight='bold')
	# step = 10
	# xticks = np.arange(0, len(data)+1, step)
	# print(np.arange(0,  400, len(xticks)//step))
	# plt.xticks(xticks, xticks*2)
	plt.yticks(np.arange(0, round(max(data))+1, (round(max(data))+1)/10))
	plt.savefig("four_layer_loss.png")
	plt.clf()


def final_word_sample():
	with open('word_sample_loss.log', 'r') as f:
		data = [float(line.split()[-1]) for line in f.readlines()[1:]]
	print(len(data))

	plt.plot(np.arange(len(data)), data)
	plt.title("Word-RNN with Sampling", fontsize=20, fontweight='bold')
	plt.xlabel("Epoch", fontweight='bold')
	plt.ylabel("Loss", fontweight='bold')
	step = 10
	xticks = np.arange(0, len(data)+1, step)
	plt.xticks(xticks, xticks*2)
	plt.yticks(np.arange(0, round(max(data))+1, (round(max(data))+1)/10))
	plt.savefig("word_sample_loss2.png")
	plt.clf()

if __name__ == '__main__':
	gru()
	four_layer()
	final_word_sample()


