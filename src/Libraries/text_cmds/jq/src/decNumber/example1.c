/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 30, 2025.
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
/* Decimal Number Library Demonstration program                       */
/* ------------------------------------------------------------------ */
/* Copyright (c) IBM Corporation, 2001, 2007.  All rights reserved.   */
/* ----------------------------------------------------------------+- */
/*                                                 right margin -->|  */

// example1.c -- convert the first two argument words to decNumber,
// add them together, and display the result

#define  DECNUMDIGITS 34           // work with up to 34 digits
#include "decNumber.h"             // base number library
#include <stdio.h>                 // for printf

int main(int argc, char *argv[]) {
  decNumber a, b;                  // working numbers
  decContext set;                  // working context
  char string[DECNUMDIGITS+14];    // conversion buffer

  decContextTestEndian(0);         // warn if DECLITEND is wrong

  if (argc<3) {                    // not enough words
    printf("Please supply two numbers to add.\n");
    return 1;
    }
  decContextDefault(&set, DEC_INIT_BASE); // initialize
  set.traps=0;                     // no traps, thank you
  set.digits=DECNUMDIGITS;         // set precision

  decNumberFromString(&a, argv[1], &set);
  decNumberFromString(&b, argv[2], &set);

  decNumberAdd(&a, &a, &b, &set);            // a=a+b
  decNumberToString(&a, string);

  printf("%s + %s => %s\n", argv[1], argv[2], string);
  return 0;
  } // main
