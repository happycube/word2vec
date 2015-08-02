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

#define AVX

#ifdef AVX
#include <immintrin.h>
#include <x86intrin.h>
#endif

#define MAX_STRING 100
#define EXP_TABLE_SIZE 512 
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 33554432;  // Maximum 33.5M * 0.7 = ~23M words in the vocabulary

typedef float real;                    // Precision of float numbers

#define MAX_SHORT_WORD 24

struct __attribute__((packed)) vocab_word {
  union __attribute__((packed)) {
	char *word;
	char shortword[MAX_SHORT_WORD];
  } w;
  long long count;
};

const long long SHORT_WORD = ((long long)1 << 62); 
const long long max_count = (((long long)1 << 62) - 1); 
struct vocab_word *vocab;

inline char * GetWordPtr(struct vocab_word *word)
{
	return (word->count & SHORT_WORD) ? word->w.shortword : word->w.word; 
}

inline char * GetWordPtrI(int index)
{
	return GetWordPtr(&vocab[index]);
}

inline long long GetWordUsage(const void *w)
{ 
	return ((struct vocab_word *)w)->count & max_count;
}

inline long long GetWordUsageI(const int i)
{ 
	return vocab[i].count & max_count;
}

inline int ReadWordIndex(FILE *fin);

struct vocab_code {
	char codelen;
	int point[MAX_CODE_LENGTH];
	char code[MAX_CODE_LENGTH];
};

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
struct vocab_word *vocab;
struct vocab_code *vocab_codes;
int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0;
// Making layer1_size const might drastically decrease # of instructions issued...
//#define CONST_LAYER1 256

#ifdef CONST_LAYER1
const long long layer1_size = CONST_LAYER1;
#else
long long layer1_size = 256;
#endif
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 1e-3;
real *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

int hs = 0, negative = 5;
//const int table_size = 1e8;
const int table_size = 134217728; // 2^27
int *table;

void InitUnigramTable() {
  int a, i;
  double train_words_pow = 0;
  double d1, power = 0.75;
  printf("table size %d\n", table_size);
  table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(GetWordUsageI(a), power);
  i = 0;
  d1 = pow(GetWordUsageI(i), power) / train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (double)table_size > d1) {
      i++;
      d1 += pow(GetWordUsageI(i), power) / train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

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
inline int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
inline int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
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

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
	unsigned int hash, length = strlen(word) + 1;
	char *word_addr = NULL;

	if (length < (MAX_SHORT_WORD - 1)) {
		vocab[vocab_size].count = 1 | SHORT_WORD;
		word_addr = vocab[vocab_size].w.shortword;
//		printf("short word %d %s\n", vocab_size, word);
	} else { 
		if (length > MAX_STRING) length = MAX_STRING;
		vocab[vocab_size].count = 1;
		vocab[vocab_size].w.word = (char *)calloc(length, sizeof(char));
		word_addr = vocab[vocab_size].w.word;
//		printf("long word %d %s\n", vocab_size, word);
	}
		
	strncpy(word_addr, word, length);
	word_addr[length - 1] = 0;
		
	// Reallocate memory if needed
	if (vocab_size + 3 >= vocab_max_size) {
		vocab_max_size += 1024;
		vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
	}

	// Find an empty hash spot.
	hash = GetWordHash(word);
	while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
	vocab_hash[hash]=vocab_size;

	return vocab_size++; // post-increment, won't actually go up until return value taken
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return GetWordUsage(b) - GetWordUsage(a);
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((GetWordUsageI(a) < min_count) && (a != 0)) {
      vocab_size--;
      if (!(vocab[a].count & SHORT_WORD)) free(vocab[a].w.word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(GetWordPtrI(a));
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += GetWordUsageI(a);
    }
  }
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  vocab_codes = (struct vocab_code *)calloc(vocab_size + 1, sizeof(struct vocab_code));
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) if (GetWordUsageI(a) > min_reduce) {
    memmove(&vocab[b], &vocab[a], sizeof(struct vocab_word));
    b++;
  } else {
     free(vocab[a].w.word);
  }
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(GetWordPtrI(a));
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree() {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  for (a = 0; a < vocab_size; a++) count[a] = GetWordUsageI(a);
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < vocab_size - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == vocab_size * 2 - 2) break;
    }
    vocab_codes[a].codelen = i;
    vocab_codes[a].point[0] = vocab_size - 2;
    for (b = 0; b < i; b++) {
      vocab_codes[a].code[i - b - 1] = code[b];
      vocab_codes[a].point[i - b] = point[b] - vocab_size;
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
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
    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word);
    if (i == -1) {
      a = AddWordToVocab(word);
    } else vocab[i].count++;
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin);
}

void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", GetWordPtrI(i), GetWordUsageI(i));
  fclose(fo);
}

#ifdef NOT // AVX
inline real DoMAC(const int n, real * __restrict__ a, real * __restrict__  b) 
{ 
	real output = 0;
	int i = 0;

	if ((!((unsigned long)a & 0x1f)) && (!((unsigned long)b & 0x1f))) {
		__m256 voutput, a1, b1, c1, a2, b2, c2;
		const float zero = 0;	
		float __attribute__ ((aligned (32))) tgt[8];
		int j = 0;

		voutput = _mm256_broadcast_ss(&zero);

		for (i = 0; i + 8 < n; i += 8) {
			a1 = _mm256_load_ps(&a[i]);
			b1 = _mm256_load_ps(&b[i]);
			c1 = _mm256_mul_ps(a1, b1); 
			voutput = _mm256_add_ps(voutput, c1);
		}
		_mm256_store_ps(tgt, voutput);
		for (j = 0; j < 8; j++) output += tgt[j];
	}	
		
	for (; i < n; i++) {
		output += a[i] * b[i]; 
	}

#if 0
	real output2 = 0;
	for (i = 0; i < n; i++) {
		output2 += a[i] * b[i]; 
	}
	if (output2 > .001) printf("%f %f\n", output, output2);
#endif
	return output;
}
#else
inline real DoMAC(const int n, real * __restrict__ a, real * __restrict__  b) 
{ 
	real output = 0;
	int i = 0;

	if ((!((unsigned long)a & 0x3f)) && (!((unsigned long)b & 0x3f))) {
		real *aa = __builtin_assume_aligned(a, 64), *ba = __builtin_assume_aligned(b, 64);
		for (i = 0; i < n; i++) output += aa[i] * ba[i];
	} else if ((!((unsigned long)a & 0x1f)) && (!((unsigned long)b & 0x1f))) {
		real *aa = __builtin_assume_aligned(a, 32), *ba = __builtin_assume_aligned(b, 32);
		for (i = 0; i < n; i++) output += aa[i] * ba[i];
	} else if ((!((unsigned long)a & 0x0f)) && (!((unsigned long)b & 0x0f))) {
		real *aa = __builtin_assume_aligned(a, 16), *ba = __builtin_assume_aligned(b, 16);
		for (i = 0; i < n; i++) output += aa[i] * ba[i];
	} else {
		for (i = 0; i < n; i++) {
			output += a[i] * b[i]; 
		}
	}
	return output;
}
#endif

#ifdef NOT // AVX
inline void DoAdd(const int n, real * __restrict__ a, real * __restrict__  b) 
{
	int i = 0;

	if ((!((unsigned long)a & 0x1f)) && (!((unsigned long)b & 0x1f))) {
		__m256 a1, b1, c1;

		for (i = 0; i + 8 < n; i += 8) {
			a1 = _mm256_load_ps(&a[i]);
			b1 = _mm256_load_ps(&b[i]);
			c1 = _mm256_add_ps(a1, b1); 
			_mm256_store_ps(&a[i], c1);
		}
	}

	for (; i < n; i++) a[i] += b[i];
}
#else
inline void DoAdd(const int n, real * __restrict__ a, real * __restrict__  b) 
{ 
	int i = 0;

	if ((!((unsigned long)a & 0x3f)) && (!((unsigned long)b & 0x3f))) {
		real *aa = __builtin_assume_aligned(a, 64), *ba = __builtin_assume_aligned(b, 64);
		for (i = 0; i < n; i++) aa[i] += ba[i];
	} else if ((!((unsigned long)a & 0x1f)) && (!((unsigned long)b & 0x1f))) {
		real *aa = __builtin_assume_aligned(a, 32), *ba = __builtin_assume_aligned(b, 32);
		for (i = 0; i < n; i++) aa[i] += ba[i];
	} else if ((!((unsigned long)a & 0x0f)) && (!((unsigned long)b & 0x0f))) {
		real *aa = __builtin_assume_aligned(a, 16), *ba = __builtin_assume_aligned(b, 16);
		for (i = 0; i < n; i++) aa[i] += ba[i];
	} else {
		for (i = 0; i < n; i++) a[i] += b[i];
	}
}
#endif

#ifdef NOT // AVX
void DoMAC1(const int n, real * __restrict__  out, real c, real * __restrict__  b) 
{ 
	int i = 0;

	if ((!((unsigned long)out & 0x1f)) && (!((unsigned long)b & 0x1f))) {
		__m256 _out, _const, _mul, _tmp;

		_const = _mm256_broadcast_ss(&c);

		for (i = 0; i + 8 < n; i += 8) {
			_out = _mm256_load_ps(&out[i]);
			_mul = _mm256_load_ps(&b[i]);
			_tmp = _mm256_mul_ps(_const, _mul); 
			_out = _mm256_add_ps(_out, _tmp);
			_mm256_store_ps(&out[i], _out);
		}
	}

	for (; i < n; i++) out[i] += c * b[i];
}
#else 
void DoMAC1(const int n, real * __restrict__  out, real c, real * __restrict__  b) 
{ 
	int i = 0;

	// On Sandy Bridge w/hyperthreading, alignment > 16 may or may not cause pipeline stalls, slowing down perf.
	// Seems to be an icache issue
/*	if ((!((unsigned long)out & 0x3f)) && (!((unsigned long)b & 0x3f))) {
		real *outa = __builtin_assume_aligned(out, 64), *ba = __builtin_assume_aligned(b, 64);
		for (i = 0; i < n; i++) outa[i] += c * ba[i];
	} else */ if ((n <= 256) && (!((unsigned long)out & 0x1f)) && (!((unsigned long)b & 0x1f))) {
		real *outa = __builtin_assume_aligned(out, 32), *ba = __builtin_assume_aligned(b, 32);
		for (i = 0; i < n; i++) outa[i] += c * ba[i];
	} else if ((!((unsigned long)out & 0x0f)) && (!((unsigned long)b & 0x0f))) {
		real *outa = __builtin_assume_aligned(out, 16), *ba = __builtin_assume_aligned(b, 16);
		for (i = 0; i < n; i++) outa[i] += c * ba[i];
	} else {
		for (i = 0; i < n; i++) out[i] += c * b[i];
	}
}
#endif

void ReadVocab() {
  long long a, i = 0;
  long long cn;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word);

    fscanf(fin, "%lld%c", &cn, &c);
    vocab[a].count = (vocab[a].count & SHORT_WORD) | cn;
    i++;
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}

  if (hs) {
    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    memset(syn1, 0, (long long)vocab_size * layer1_size * sizeof(real));
  }

  if (negative>0) {
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    memset(syn1neg, 0, (long long)vocab_size * layer1_size * sizeof(real));
  }

  for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) {
    next_random = (next_random + 11) * (unsigned long long)25214903917;
    syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
  }

  CreateBinaryTree();
}

const real EXP_SCALE = (real)EXP_TABLE_SIZE / (real)MAX_EXP;

inline real getExp(real r)
{
	real rabs = fabs(r), rv = 0.5;

	if (rabs < MAX_EXP) {
		rv = expTable[(int)(rabs * EXP_SCALE)];
	}

	rv = (r > 0) ? (0.5 + rv) : (0.5 - rv);

	return rv;
}

void *TrainModelThread(void *id) {
  long long a, b, d, cw, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, c, target, label, local_iter = iter;
  unsigned long long next_random = (long long)id;
  real f, g;
  clock_t now;

  real *neu1;
  a = posix_memalign((void **)&neu1, 128, layer1_size * sizeof(real));
  real *neu1e; // = (real *)calloc(layer1_size, sizeof(real));
  a = posix_memalign((void **)&neu1e, 128, layer1_size * sizeof(real));

  FILE *fi = fopen(train_file, "rb");

  memset(sen, 0, sizeof(sen));

  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now=clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
         word_count_actual / (real)(iter * train_words + 1) * 100,
         word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
    if (sentence_length == 0) {
      while (1) {
	real ran;
        word = ReadWordIndex(fi);
        if (feof(fi)) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0) break;
        // The subsampling randomly discards frequent words while keeping the ranking same
        if (sample > 0) {
          next_random = (next_random + 11) * (unsigned long long)25214903917;
          ran = (sqrt(GetWordUsageI(word) / (sample * train_words)) + 1) * (sample * train_words) / GetWordUsageI(word);
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }

    if (feof(fi) || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
      continue;
    }

    word = sen[sentence_position];
    if (word == -1) continue;
    for (c = 0; c < layer1_size; c++) neu1[c] = neu1e[c] = 0;
    b = next_random % window;
    next_random = (next_random + 11) * (unsigned long long)25214903917;

    if (cbow) {  //train the cbow architecture
//	struct vocab_word *vocword = &vocab[word];
	struct vocab_code *voccode = &vocab_codes[word];
      // in -> hidden
      cw = 0;
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if ((c < 0) || (c >= sentence_length)) continue;

        last_word = sen[c];
        if (last_word == -1) continue;

        for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size];

        cw++;
      }

      if (cw) {
        for (c = 0; c < layer1_size; c++) neu1[c] /= cw;
        if (hs) for (d = 0; d < voccode->codelen; d++) {
          f = 0;
          l2 = voccode->point[d] * layer1_size;

          real *syn1_l2 = &syn1[l2]; 

          // Propagate hidden -> output
          //for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1_l2[c];
	f = DoMAC(layer1_size, neu1, syn1_l2);

	f = getExp(f);

          // 'g' is the gradient multiplied by the learning rate
          g = (1 - voccode->code[d] - f) * alpha;
	
	// Propagate errors output -> hidden
	DoMAC1(layer1_size, neu1e, g, syn1_l2);
        // Learn weights hidden -> output
	DoMAC1(layer1_size, syn1_l2, g, neu1);
        }

        // NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
	  register long long next_target;
          if (d == 0) {
            target = word;
            label = 1;
            next_target = table[(next_random >> 16) % table_size];
            next_random = (next_random + 11) * (unsigned long long)25214903917;
          } else {
            next_target = table[(next_random >> 16) % table_size];
            next_random = (next_random + 11) * (unsigned long long)25214903917;
            
	    if (target == 0) target = next_random % (vocab_size - 1) + 1;

            if (target == word) {
	  	target = next_target;
		continue;
            }
            label = 0;
          }

          l2 = target * layer1_size;
          real *syn1neg_l2 = &syn1neg[l2]; 

	  f = DoMAC(layer1_size, neu1, syn1neg_l2);

	g = (label - getExp(f)) * alpha;

	DoMAC1(layer1_size, neu1e, g, syn1neg_l2);
	DoMAC1(layer1_size, syn1neg_l2, g, neu1);

	target = next_target;
        }

        // hidden -> in
        for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
          c = sentence_position - window + a;
          if ((c < 0) || (c >= sentence_length)) continue;

          last_word = sen[c];
          if (last_word == -1) continue;

          for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];
        }
      }
    } else {  //train skip-gram
      register unsigned long long _next_random = next_random;
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if ((c < 0) || (c >= sentence_length)) continue;
	
        last_word = sen[c];
        if (last_word == -1) continue;

        l1 = last_word * layer1_size;
        real *syn0_l1 = &syn0[l1]; 

        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;

        // HIERARCHICAL SOFTMAX
        if (hs) {
	 struct vocab_code *voccode = &vocab_codes[word];
  	 for (d = 0; d < voccode->codelen; d++) {
          l2 = voccode->point[d] * layer1_size;
          real *syn1_l2 = &syn1[l2]; 

          // Propagate hidden -> output
	  f = DoMAC(layer1_size, syn0_l1, syn1_l2);

	  f = getExp(f);

          // 'g' is the gradient multiplied by the learning rate
          g = (1 - voccode->code[d] - f) * alpha;
	  DoMAC1(layer1_size, neu1e, g, syn1_l2);
	  DoMAC1(layer1_size, syn1_l2, g, syn0_l1);
        }
       }
        // NEGATIVE SAMPLING
        if (negative > 0) {
         register long long next_target;
	 for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word;
            label = 1;
            next_target = table[(_next_random >> 16) % table_size];
            _next_random = (_next_random + 11) * (unsigned long long)25214903917;
          } else {
            next_target = table[(_next_random >> 16) % table_size];
            _next_random = (_next_random + 11) * (unsigned long long)25214903917;

            if (target == 0) target = _next_random % (vocab_size - 1) + 1;
            if (target == word) {
	  	target = next_target;
		continue;
            }
            label = 0;
          }
          l2 = target * layer1_size;
          real *syn1neg_l2 = &syn1neg[l2]; 

	  f = DoMAC(layer1_size, syn0_l1, syn1neg_l2);
      
	  g = (label - getExp(f)) * alpha;
	  DoMAC1(layer1_size, neu1e, g, syn1neg_l2);
	  DoMAC1(layer1_size, syn1neg_l2, g, syn0_l1);
	  target = next_target;
        }
        // Learn weights input -> hidden
	DoAdd(layer1_size, syn0_l1, neu1e);
//        for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
       }
      }
      next_random = _next_random + 11;
    }
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
  pthread_exit(NULL);
}

void TrainModel() {
  long a, b, c, d;
  FILE *fo;
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;
  if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
  if (save_vocab_file[0] != 0) SaveVocab();
  if (output_file[0] == 0) return;
  InitNet();
  if (negative > 0) InitUnigramTable();
  start = clock();
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
  fo = fopen(output_file, "wb");
  if (classes == 0) {
    // Save the word vectors
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", GetWordPtrI(a));
      if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
      else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
      fprintf(fo, "\n");
    }
  } else {
    // Run K-means on the word vectors
    int clcn = classes, iter = 10, closeid;
    int *centcn = (int *)malloc(classes * sizeof(int));
    int *cl = (int *)calloc(vocab_size, sizeof(int));
    real closev, x;
    real *cent = (real *)calloc(classes * layer1_size, sizeof(real));

    for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;

    for (a = 0; a < iter; a++) {
      for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
      for (b = 0; b < clcn; b++) centcn[b] = 1;

      for (c = 0; c < vocab_size; c++) {
        for (d = 0; d < layer1_size; d++) cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
        centcn[cl[c]]++;
      }

      for (b = 0; b < clcn; b++) {
        closev = 0;
        for (c = 0; c < layer1_size; c++) {
          cent[layer1_size * b + c] /= centcn[b];
          closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
        }
        closev = sqrt(closev);
        for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
      }

      for (c = 0; c < vocab_size; c++) {
        closev = -10;
        closeid = 0;
        for (d = 0; d < clcn; d++) {
          x = 0;
          for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
          if (x > closev) {
            closev = x;
            closeid = d;
          }
        }

        cl[c] = closeid;
      }
    }
    // Save the K-means classes
    for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", GetWordPtrI(a), cl[a]);

    free(centcn);
    free(cent);
    free(cl);
  }
  fclose(fo);
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
    printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 12)\n");
    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations (default 5)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-cbow <int>\n");
    printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
    return 0;
  }
  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
#ifndef CONST_LAYER1
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
#endif
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  if (cbow) alpha = 0.05;
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);

  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));

  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 1 - 0) * MAX_EXP); // Precompute the exp() table
    expTable[i] = (expTable[i] / (expTable[i] + 1)) - 0.5;                   // Precompute f(x) = x / (x + 1)
  }

  TrainModel();
  return 0;
}