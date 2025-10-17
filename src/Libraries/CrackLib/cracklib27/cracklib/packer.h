/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 8, 2024.
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
#include <stdio.h>
#include <ctype.h>
#include <stdint.h>

#define STRINGSIZE		1024
#define TRUNCSTRINGSIZE		(STRINGSIZE/4)

typedef uint8_t int8;
typedef uint16_t int16;
typedef uint32_t int32;
#ifndef NUMWORDS
#define NUMWORDS 	16
#endif
#define MAXWORDLEN	32
#define MAXBLOCKLEN 	(MAXWORDLEN * NUMWORDS)

struct pi_header
{
    int32 pih_magic;
    int32 pih_numwords;
    int16 pih_blocklen;
    int16 pih_pad;
};

typedef struct
{
    FILE *ifp;
    FILE *dfp;
    FILE *wfp;

    int32 flags;
#define PFOR_WRITE	0x0001
#define PFOR_FLUSH	0x0002
#define PFOR_USEHWMS	0x0004

    int32 hwms[256];

    struct pi_header header;

    int count;
    char data[NUMWORDS][MAXWORDLEN];
} PWDICT;

#define PW_WORDS(x) ((x)->header.pih_numwords)
#define PIH_MAGIC 0x70775631

extern PWDICT *PWOpen();
extern char *Mangle();
extern char *FascistCheck();

#define CRACK_TOLOWER(a) 	(isupper(a)?tolower(a):(a)) 
#define CRACK_TOUPPER(a) 	(islower(a)?toupper(a):(a)) 
#define STRCMP(a,b)		strcmp((a),(b))
