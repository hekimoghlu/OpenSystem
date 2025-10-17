/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 4, 2023.
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
/* Copyright (c) IBM Corporation, 2001, 2008.  All rights reserved.   */
/* ----------------------------------------------------------------+- */
/*                                                 right margin -->|  */

// example7.c -- using decQuad to add two numbers together

// compile: example7.c decContext.c decQuad.c

#include "decQuad.h"               // decQuad library
#include <stdio.h>                 // for printf

int main(int argc, char *argv[]) {
  decQuad a, b;                    // working decQuads
  decContext set;                  // working context
  char string[DECQUAD_String];     // number->string buffer

  decContextTestEndian(0);         // warn if DECLITEND is wrong

  if (argc<3) {                    // not enough words
    printf("Please supply two numbers to add.\n");
    return 1;
    }
  decContextDefault(&set, DEC_INIT_DECQUAD); // initialize

  decQuadFromString(&a, argv[1], &set);
  decQuadFromString(&b, argv[2], &set);
  decQuadAdd(&a, &a, &b, &set);    // a=a+b
  decQuadToString(&a, string);

  printf("%s + %s => %s\n", argv[1], argv[2], string);
  return 0;
  } // main
