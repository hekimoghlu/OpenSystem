/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 29, 2023.
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

// example5.c -- decimal64 conversions

#include "decimal64.h"             // decimal64 and decNumber library
#include <stdio.h>                 // for (s)printf

int main(int argc, char *argv[]) {
  decimal64 a;                     // working decimal64 number
  decNumber d;                     // working number
  decContext set;                  // working context
  char string[DECIMAL64_String];   // number->string buffer
  char hexes[25];                  // decimal64->hex buffer
  int i;                           // counter

  if (argc<2) {                    // not enough words
    printf("Please supply a number.\n");
    return 1;
    }
  decContextDefault(&set, DEC_INIT_DECIMAL64); // initialize

  decimal64FromString(&a, argv[1], &set);
  // lay out the decimal64 as eight hexadecimal pairs
  for (i=0; i<8; i++) {
    sprintf(&hexes[i*3], "%02x ", a.bytes[i]);
    }
  decimal64ToNumber(&a, &d);
  decNumberToString(&d, string);
  printf("%s => %s=> %s\n", argv[1], hexes, string);
  return 0;
  } // main
