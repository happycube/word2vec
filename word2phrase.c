//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 60
#define MAX_SHORT_WORD 12

const int vocab_hash_size = 536870912; // Maximum 2^29 entries in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
  unsigned int cn;
  union {
	char *word;
	char shortword[MAX_SHORT_WORD];
  } w;
};

const unsigned int SHORT_WORD = (1 << 31); 
const unsigned int max_count = ((unsigned int)(1 << 31) - 1); 
struct vocab_word *vocab;

inline char * GetWordPtr(struct vocab_word *word)
{
	return (word->cn & SHORT_WORD) ? word->w.shortword : word->w.word; 
}

inline char * GetWordPtrI(int index)
{
	return GetWordPtr(&vocab[index]);
}

inline int GetWordUsage(const void *w)
{ 
	return ((struct vocab_word *)w)->cn & max_count;
}

inline int GetWordUsageI(const int i)
{ 
	return vocab[i].cn & max_count;
}

char train_file[MAX_STRING], output_file[MAX_STRING];
int debug_mode = 2, min_count = 5, *vocab_hash, min_reduce = 1;
long long vocab_max_size = 32768, vocab_size_increment = 32768, vocab_size = 0;
long long train_words = 0;
real threshold = 100;

unsigned long long next_random = 1;

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 1;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
//    printf("%x %d %x %p %s\n", hash, vocab_hash[hash], vocab[vocab_hash[hash]].cn, GetWordPtrI(vocab_hash[hash]), GetWordPtrI(vocab_hash[hash]));
    if (!strcmp(word, GetWordPtrI(vocab_hash[hash]))) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// For accounting and tracking how many words got stored in the structure
long long words = 0, short_words = 0;

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
	unsigned int hash, length = strlen(word) + 1;

	words++;
	if (length <= MAX_SHORT_WORD) {
		short_words++;
		strncpy(vocab[vocab_size].w.shortword, word, length);
		vocab[vocab_size].cn = 1 | SHORT_WORD;
//		printf("short word %d %s\n", vocab_size, word);
	} else { 
		if (length > MAX_STRING) length = MAX_STRING;
		vocab[vocab_size].w.word = (char *)calloc(length, sizeof(char));
		vocab[vocab_size].cn = 1;
//		printf("long word %d %s\n", vocab_size, word);
	}
		
	strncpy(GetWordPtrI(vocab_size), word, length);

	// Reallocate memory if needed
	if (vocab_size + 3 >= vocab_max_size) {
		vocab_max_size += vocab_size_increment;
		vocab=(struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
	}

	// Find an empty hash spot.
	hash = GetWordHash(word);
	while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
	vocab_hash[hash]=vocab_size;

	return vocab_size++; // post-increment, won't actually go up until return value taken
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
//    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
	return GetWordUsage(b) - GetWordUsage(a);
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab(int min_usage) {
	int a, new_vocab_size = 0;
	unsigned int hash;

	for (a = 0; a < vocab_size; a++) { 
		if (GetWordUsageI(a) > min_usage) {
			vocab[new_vocab_size].cn = vocab[a].cn;
			if (a != new_vocab_size) memcpy(&vocab[new_vocab_size].w, &vocab[a].w, sizeof(vocab[a].w));
			new_vocab_size++;
		} else {
			if (!(vocab[a].cn & SHORT_WORD)) free(vocab[a].w.word);
		}
	}

	vocab_size = new_vocab_size;

	// Recompute hashes, since multiple words may have hashed the same way 
	for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
	for (a = 0; a < vocab_size; a++) {
		hash = GetWordHash(GetWordPtrI(a));
		while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
		vocab_hash[hash] = a;
	}
	fflush(stdout);
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
	// Sort the vocabulary and keep </s> at the first position
	qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
}

void LearnVocabFromTrainFile() {
  char word[MAX_STRING], last_word[MAX_STRING], bigram_word[MAX_STRING * 2];
  FILE *fin;
  long long a, i, start = 1;

  strcpy(word, "");
  strcpy(last_word, "");

  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *)"</s>");
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    if (!strcmp(word, "</s>")) {
      start = 1;
      continue;
    } else start = 0;
    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("Words processed: %lldK     Vocab size: %lldK  %c", train_words / 1000, vocab_size / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word);
    //printf("%d %s\n", i, word);
    if (i == -1) {
      a = AddWordToVocab(word);
    } else {
	if (GetWordUsageI(i) < max_count) vocab[i].cn++;
    }
    if (start) continue;
    sprintf(bigram_word, "%s_%s", last_word, word);
    bigram_word[MAX_STRING - 1] = 0;
    strcpy(last_word, word);
    i = SearchVocab(bigram_word);
    if (i == -1) {
      a = AddWordToVocab(bigram_word);
    } else {
	if (GetWordUsageI(i) < max_count) vocab[i].cn++;
    }
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab(min_reduce++);
  }
  SortVocab();
  ReduceVocab(min_count);
  if (debug_mode > 0) {
    printf("\nVocab size (unigrams + bigrams): %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fclose(fin);
}

void TrainModel() {
  long long pa = 0, pb = 0, pab = 0, oov, i, li = -1, cn = 0;
  char word[MAX_STRING], last_word[MAX_STRING], bigram_word[MAX_STRING * 2];
  real score;
  FILE *fo, *fin;
  printf("Starting training using file %s\n", train_file);
  LearnVocabFromTrainFile();
  fin = fopen(train_file, "rb");
  fo = fopen(output_file, "wb");
  word[0] = 0;
  while (1) {
    strcpy(last_word, word);
    ReadWord(word, fin);
    if (feof(fin)) break;
    if (!strcmp(word, "</s>")) {
      fprintf(fo, "\n");
      continue;
    }
    cn++;
    if ((debug_mode > 1) && (cn % 100000 == 0)) {
      printf("Words written: %lldK%c", cn / 1000, 13);
      fflush(stdout);
    }
    oov = 0;
    i = SearchVocab(word);
    if (i == -1) oov = 1; else pb = GetWordUsageI(i);
    if (li == -1) oov = 1;
    li = i;
    sprintf(bigram_word, "%s_%s", last_word, word);
    bigram_word[MAX_STRING - 1] = 0;
    i = SearchVocab(bigram_word);
    if (i == -1) oov = 1; else pab = GetWordUsageI(i);
    if (pa < min_count) oov = 1;
    if (pb < min_count) oov = 1;
    if (oov) score = 0; else score = (pab - min_count) / (real)pa / (real)pb * (real)train_words;
    if (score > threshold) {
      fprintf(fo, "_%s", word);
      pb = 0;
    } else fprintf(fo, " %s", word);
    pa = pb;
  }
  fclose(fo);
  fclose(fin);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD2PHRASE tool v0.1a\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters / phrases\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-threshold <float>\n");
    printf("\t\t The <float> value represents threshold for forming the phrases (higher means less phrases); default 100\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\nExamples:\n");
    printf("./word2phrase -train text.txt -output phrases.txt -threshold 100 -debug 2\n\n");
    return 0;
  }
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threshold", argc, argv)) > 0) threshold = atof(argv[i + 1]);
  printf("%d\n", sizeof(struct vocab_word));
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  TrainModel();
  printf("%lld %lld %x %d\n", words, short_words, max_count, sizeof(unsigned int)); 
  return 0;
}
