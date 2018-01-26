import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os
from langdetect import detect
import langid
plt.style.use("seaborn")
plt.rcParams['figure.dpi'] = 300


def most_common_words():
	with open('output3.txt', 'r') as f:
		data = f.readlines()

	new_lines = len(data)
	data = [line.split() for line in data]
	data = [word for sentence in data for word in sentence]
	counter = Counter(data)
	total_words = len(data) 
	print(total_words)
	print(len(counter))
	print(new_lines)

	num=15
	print(counter.most_common(num))

	top = counter.most_common(num)
	top_words = [pair[0] for pair in top]
	top_counts = np.array([pair[1] for pair in top])

	fig, ax = plt.subplots()
	ax.barh(np.arange(num), top_counts/total_words, align='center',
	        ecolor='black')
	ax.set_yticks(np.arange(num))
	ax.set_yticklabels(top_words)
	ax.invert_yaxis()  # labels read top-to-bottom
	ax.set_xlabel('Rate of Occurrence')
	ax.set_title('Most Common Words', fontsize=20, fontweight='bold')
	plt.savefig("graphics/top_words.png")

def most_common_chars():
	with open('output3.txt', 'r') as f:
		data = f.read().split()

	data = [char for word in data for char in word]
	counter = Counter(data)
	total_words = len(data) 
	print(total_words)

	# num=len(counter)
	# print(counter.most_common(num))

	# top = counter.most_common(num)
	counts = list(counter.values())
	chars = list(counter.keys())

	num = 27
	graph_counts, graph_chars = zip(*sorted(zip(counts[:num], chars[:num])))

	print(len(counts))
	print(len(counter))
	print(chars[:num])
	print(sum(graph_counts))

	fig, ax = plt.subplots()
	ax.barh(np.arange(num), np.array(graph_counts[::-1])/total_words, align='center', ecolor='black')
	ax.invert_yaxis()  # labels read top-to-bottom
	ax.set_yticks(np.arange(num))
	ax.set_yticklabels(graph_chars[::-1])
	ax.set_xlabel('Rate of Occurrence')
	ax.set_title('Character Counts', fontsize=20, fontweight='bold')
	plt.savefig("graphics/top_chars.png")

def sentence_length():
	# SENTENCE LENGTH HISTOGRAM
	with open('output3.txt', 'r') as f:
		data = f.readlines()

	new_lines = len(data)
	sentence_lengths = [len(line.split()) for line in data if len(line.split()) > 1]
	sentence_lengths.sort()
	print(max(sentence_lengths))
	print(sentence_lengths[-5:])
	outliers = sum(1 for i in sentence_lengths if i >= 40)
	print(outliers)
	print(outliers/new_lines)
	fig, ax = plt.subplots()
	ax.hist(sentence_lengths[:-outliers], normed=True, bins=25)
	ax.set_xlabel('Number of Words')
	ax.set_ylabel('Proportion')
	ax.set_title('Distribution of Line Lengths', fontsize=20, fontweight='bold')
	plt.savefig("graphics/line_lengths.png")

def poem_lines():
	poems = []
	count = []
	for file in os.listdir("original_poems")[1:]:
		with open("original_poems/" + file, 'r') as f:
			lines = len(f.readlines())
			if lines < 200:
				poems.append(lines)
			else:
				count.append(lines)
	total_poems = len(poems) + len(count)
	print(total_poems)
	print(len(count)/total_poems)
	print(count)
	fig, ax = plt.subplots()
	ax.hist(poems, normed=True, bins=25)
	ax.set_xlabel('Number of Lines')
	ax.set_ylabel('Density')
	ax.set_title('Distribution of Lines in a Poem', fontsize=20, fontweight='bold')
	plt.savefig("graphics/poem_lines.png")

def poem_words():
	poems = []
	count = []
	for file in os.listdir("original_poems")[1:]:
		with open("original_poems/" + file, 'r') as f:
			words = len(f.read().split())
			if words < 2000:
				poems.append(words)
			else:
				count.append(words)
	total_poems = len(poems) + len(count)
	print(total_poems)
	print(len(count)/total_poems)
	print(count)
	fig, ax = plt.subplots()
	ax.hist(poems, normed=True, bins=25)
	ax.set_xlabel('Number of Words')
	ax.set_ylabel('Density')
	ax.set_title('Distribution of Number of Words in a Poem', fontsize=20, fontweight='bold')
	plt.savefig("graphics/poem_words.png")

def languages():
	langs = []
	for file in os.listdir("original_poems")[1:]:
		with open("original_poems/" + file, 'r') as f:
			poem = f.read()
			try:
				langs.append(langid.classify(poem)[0])
			except:
				print(poem)
	print(len(langs))
	print(langs[0])
	fig, ax = plt.subplots()
	counter = Counter(langs)
	# print(counter)
	print(counter.values())


if __name__ == '__main__':
	# most_common_words()
	most_common_chars()
	# sentence_length
	# poem_lines()
	# poem_words()
	# languages()


