/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 26, 2024.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#include "xotclInt.h"

char *
XOTcl_ltoa(char *buf, long i, int *len)  /* fast version of sprintf(buf,"%ld",l); */ {
  int nr_written, negative;
  char tmp[LONG_AS_STRING], *pointer = &tmp[1], *string, *p;
  *tmp = 0;
  
  if (i<0) {
    i = -i;
    negative = nr_written = 1;
  } else 
    nr_written = negative = 0;
  
  do {
    nr_written++;
    *pointer++ = i%10 + '0';
    i/=10;
  } while (i);
  
  p = string = buf;
  if (negative)
    *p++ = '-';
  
  while ((*p++ = *--pointer));   /* copy number (reversed) from tmp to buf */
  if (len) *len = nr_written;
  return string;
}


static char *alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
static int blockIncrement = 8;
/*
static char *alphabet = "ab";
static int blockIncrement = 2;
*/
static unsigned char chartable[255] = {0};


char *
XOTclStringIncr(XOTclStringIncrStruct *iss) {
  char newch, *currentChar;

  currentChar = iss->buffer + iss->bufSize - 2;
  newch = *(alphabet + chartable[(unsigned)*currentChar]);
    
  while (1) {
    if (newch) { /* no overflow */
      *currentChar = newch;
      break;
    } else {     /* overflow */
      *currentChar = *alphabet; /* use first char from alphabet */
      currentChar--;
      assert(currentChar >= iss->buffer);

      newch = *(alphabet + chartable[(unsigned)*currentChar]);
      if (currentChar < iss->start) {
	iss->length++;
	if (currentChar == iss->buffer) {
	  size_t newBufSize = iss->bufSize + blockIncrement;
	  char *newBuffer = ckalloc(newBufSize);
	  currentChar = newBuffer+blockIncrement;
	  /*memset(newBuffer, 0, blockIncrement);*/
	  memcpy(currentChar, iss->buffer, iss->bufSize);
	  *currentChar = newch;
	  iss->start = currentChar;
	  ckfree(iss->buffer);
	  iss->buffer = newBuffer;
	  iss->bufSize = newBufSize;
	} else {
	  iss->start = currentChar;
	}
      }
    }
  }
  assert(iss->buffer[iss->bufSize-1] == 0);
  assert(iss->buffer[iss->bufSize-2] != 0);
  assert(iss->length < iss->bufSize);
  assert(iss->start + iss->length + 1 == iss->buffer + iss->bufSize);

  return iss->start;
}


void
XOTclStringIncrInit(XOTclStringIncrStruct *iss) {
  char *p;
  int i = 0;
  const size_t bufSize = blockIncrement>2 ? blockIncrement : 2;

  for (p=alphabet; *p; p++) {
    chartable[(int)*p] = ++i;
  }

  iss->buffer = ckalloc(bufSize);
  memset(iss->buffer, 0, bufSize);
  iss->start    = iss->buffer + bufSize-2;
  iss->bufSize  = bufSize;
  iss->length   = 1;
  /*
    for (i=1; i<50; i++) {
      XOTclStringIncr(iss);
      fprintf(stderr, "string '%s' (%d)\n",  iss->start, iss->length);
    }
  */
}

void
XOTclStringIncrFree(XOTclStringIncrStruct *iss) {
  ckfree(iss->buffer);
}
