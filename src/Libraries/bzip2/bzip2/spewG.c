/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 1, 2022.
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
/* ------------------------------------------------------------------
   This file is part of bzip2/libbzip2, a program and library for
   lossless, block-sorting data compression.

   bzip2/libbzip2 version 1.0.8 of 13 July 2019
   Copyright (C) 1996-2019 Julian Seward <jseward@acm.org>

   Please read the WARNING, DISCLAIMER and PATENTS sections in the 
   README file.

   This program is released under the terms of the license contained
   in the file LICENSE.
	 ------------------------------------------------------------------ */


#define _FILE_OFFSET_BITS 64

#include <stdio.h>
#include <stdlib.h>

/* The number of megabytes of junk to spew out (roughly) */
#define MEGABYTES 5000

#define N_BUF 1000000
char buf[N_BUF];

int main ( int argc, char** argv )
{
   int ii, kk, p;
   srandom(1);
   setbuffer ( stdout, buf, N_BUF );
   for (kk = 0; kk < MEGABYTES * 515; kk+=3) {
      p = 25+random()%50;
      for (ii = 0; ii < p; ii++)
         printf ( "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" );
      for (ii = 0; ii < p-1; ii++)
         printf ( "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb" );
      for (ii = 0; ii < p+1; ii++)
         printf ( "ccccccccccccccccccccccccccccccccccccc" );
   }
   fflush(stdout);
   return 0;
}
