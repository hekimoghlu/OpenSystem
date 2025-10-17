/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 20, 2025.
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
/* Copyright (c) IBM Corporation, 2001.  All rights reserved.         */
/* ----------------------------------------------------------------+- */
/*                                                 right margin -->|  */

// example6.c -- calculate compound interest, using Packed Decimal
// Values are investment, rate (%), and years

#include "decPacked.h"             // base number library
#include <stdio.h>                 // for printf

int main(int argc, char *argv[]) {
  { // excerpt for User's Guide starts here--------------------------|
  decNumber one, mtwo, hundred;                   // constants
  decNumber start, rate, years;                   // parameters
  decNumber total;                                // result
  decContext set;                                 // working context

  uint8_t startpack[]={0x01, 0x00, 0x00, 0x0C};   // investment=100000
  int32_t startscale=0;
  uint8_t ratepack[]={0x06, 0x5C};                // rate=6.5%
  int32_t ratescale=1;
  uint8_t yearspack[]={0x02, 0x0C};               // years=20
  int32_t yearsscale=0;
  uint8_t respack[16];                            // result, packed
  int32_t resscale;                               // ..
  char  hexes[49];                                // for packed->hex
  int   i;                                        // counter

  if (argc<0) printf("%s", argv[1]);              // noop for warning

  decContextDefault(&set, DEC_INIT_BASE);         // initialize
  set.traps=0;                                    // no traps
  set.digits=25;                                  // precision 25
  decNumberFromString(&one,       "1", &set);     // set constants
  decNumberFromString(&mtwo,     "-2", &set);
  decNumberFromString(&hundred, "100", &set);

  decPackedToNumber(startpack, sizeof(startpack), &startscale, &start);
  decPackedToNumber(ratepack,  sizeof(ratepack),  &ratescale,  &rate);
  decPackedToNumber(yearspack, sizeof(yearspack), &yearsscale, &years);

  decNumberDivide(&rate, &rate, &hundred, &set);  // rate=rate/100
  decNumberAdd(&rate, &rate, &one, &set);         // rate=rate+1
  decNumberPower(&rate, &rate, &years, &set);     // rate=rate^years
  decNumberMultiply(&total, &rate, &start, &set); // total=rate*start
  decNumberRescale(&total, &total, &mtwo, &set);  // two digits please

  decPackedFromNumber(respack, sizeof(respack), &resscale, &total);

  // lay out the total as sixteen hexadecimal pairs
  for (i=0; i<16; i++) {
    sprintf(&hexes[i*3], "%02x ", respack[i]);
    }
  printf("Result: %s (scale=%ld)\n", hexes, (long int)resscale);

  } //---------------------------------------------------------------|
  return 0;
  } // main
