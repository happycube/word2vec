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
#include <string.h>
#include <math.h>
#include <malloc.h>

const long long max_size = 2000;         // max length of strings
const long long N = 40;                  // number of closest words that will be shown
const long long max_w = 60;              // max length of vocabulary entries

int main(int argc, char **argv) {
  FILE *f;
  char st1[max_size];
  long long bestw[N];
  char file_name[max_size], st[100][max_size];
  float dist, len, bestd[N], vec[max_size];
  long long words, size, a, b, c, cn, bi[100];
  float **M;
  char **vocab;
  float *norm;
  if (argc < 2) {
    printf("Usage: ./distance <FILE>\nwhere FILE contains word projections in the BINARY FORMAT\n");
    return 0;
  }
  strcpy(file_name, argv[1]);
  f = fopen(file_name, "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    return -1;
  }

  // Read the # of words and the vector length/size per word.
  fscanf(f, "%lld", &words);
  fscanf(f, "%lld", &size);

  // Allocate the vocabulary 2-dimensional array
  vocab = (char * *)malloc((long long)words * sizeof(char *));
  for (a = 0; a < words; a++) vocab[a] = (char *)malloc(max_w * sizeof(char));

  // Allocate normalization values and vec arrays  
  norm = (float *)malloc((long long)words * sizeof(float));
  M = (float * *)malloc((long long)words * sizeof(float *));
  // XXX: not checking for proper allocations, since this is temp code
  for (a = 0; a < words; a++) M[a] = (float *)malloc(size * sizeof(float));

  // Load the words and vectors 
  for (b = 0; b < words; b++) {
    a = 0;
    while (1) {
      vocab[b][a] = fgetc(f);
      if (feof(f) || (vocab[b][a] == ' ') || (a == max_w - 1)) break;
      if (vocab[b][a] != '\n') a++;
    }
    vocab[b][a] = 0;

    // Load the vector array
    fread(&M[b][0], sizeof(float), size, f);
    len = 0;
    for (; a < size; a++) {
        len += (M[b][a] * M[b][a]);
    }
    norm[b] = sqrt(len); // Save the normalization value
  }
  fclose(f);

  // Main loop
  while (1) {
    printf("Enter word or sentence (EXIT to break): ");
    a = 0;
    while (1) {
      int rv = fgetc(stdin);

      if (rv == EOF) return(0);
      st1[a] = rv;
      if ((st1[a] == '\n') || (a >= max_size - 1)) {
        st1[a] = 0;
        break;
      }
      a++;
    }
    if (!strcmp(st1, "EXIT")) break;
    cn = 0;
    b = 0;
    c = 0;
    while (1) {
      st[cn][b] = st1[c];
      b++;
      c++;
      st[cn][b] = 0;
      if (st1[c] == 0) break;
      if (st1[c] == ' ') {
        cn++;
        b = 0;
        c++;
      }
    }
    cn++;

    // Scan the dictionary for each input word
    for (a = 0; a < cn; a++) {
      for (b = 0; b < words; b++) if (!strcmp(vocab[b], st[a])) break;
      if (b == words) b = -1;
      bi[a] = b;
      printf("\nWord: %s  Position in vocabulary: %lld\n", st[a], bi[a]);
      if (b == -1) {
        printf("Out of dictionary word!\n");
        break;
      }
    }
    if (b == -1) continue;
    printf("\n                                              Word       Cosine distance\n------------------------------------------------------------------------\n");

	// Add the vectors of each selected word
    for (a = 0; a < size; a++) vec[a] = 0;
    for (b = 0; b < cn; b++) {
      if (bi[b] == -1) continue;
      for (a = 0; a < size; a++) vec[a] += M[bi[b]][a];
    }

	// Normalize the summed vector
    len = 0;
    for (a = 0; a < size; a++) len += vec[a] * vec[a];
    len = sqrt(len);
    for (a = 0; a < size; a++) vec[a] /= len;

	// Clear the list of best matches
    for (a = 0; a < N; a++) {
	bestd[a] = bestw[a] = -1;
    }

	// Now do the actual comparison work
	for (c = 0; c < words; c++) {
		// Skip any words contained in our phrase
		a = 0;
		for (b = 0; b < cn; b++) if (bi[b] == c) a = 1;
		if (a == 1) continue;

		// Take the product of our phrase and the target word
		dist = 0;
		for (a = 0; a < size; a++) dist += vec[a] * M[c][a]; 
		dist /= norm[c]; // Normalize

		// If this is one of the best matches, put it in the list
		if (dist > bestd[N - 1]) {
			for (a = 0; (a < N); a++) {
			if (dist > bestd[a]) {
				memmove(&bestd[a + 1], &bestd[a], sizeof(float) * (N - a - 1));
				memmove(&bestw[a + 1], &bestw[a], sizeof(long long) * (N - a - 1));

				bestd[a] = dist;
				bestw[a] = c;
				break;
			}
		}
	}
    }
    // All done!  Now print the list
    for (a = 0; a < N; a++) {
	if (bestw[a] >= 0) printf("%50s\t\t%f\n", vocab[bestw[a]], bestd[a]);
    }
  }
  return 0;
}
