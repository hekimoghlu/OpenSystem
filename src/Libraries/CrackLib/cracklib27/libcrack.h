/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 17, 2022.
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

#ifdef USE_PWD_H
#include <pwd.h>
#endif	/* USE_PWD_H */

#ifdef USE_SYS_TYPES_H
#include <sys/types.h>
#endif	/* USE_SYS_TYPES_H */

#ifdef USE_SIGNAL_H
#include <signal.h>
#endif /* USE_SIGNAL_H */

#ifdef USE_STRING_H
#include <string.h>
#endif /* USE_STRING_H */

#ifdef USE_STRINGS_H
#include <strings.h>
#endif /* USE_STRINGS_H */

#ifdef USE_STDLIB_H
#include <stdlib.h>
#endif /* USE__H */

#ifdef USE_UNISTD_H
#include <unistd.h>
#endif	/* USE_UNISTD_H */

#ifdef USE_MALLOC_H
#include <malloc.h>
#endif /* USE_MALLOC_H */

#ifdef USE_BZERO
#define ClrMem(addr, cnt)	bzero(addr, cnt)
#else /* USE_BZERO */
#define ClrMem(addr, cnt)	memset(addr, '\0', cnt)
#endif /* USE_BZERO */

#ifndef NUMWORDS
#define NUMWORDS 	        16
#endif /* NUMWORDS */

#ifndef MAXWORDLEN
#define MAXWORDLEN	        32
#endif /* MAXWORDLEN */

#define MAXBLOCKLEN 	        (MAXWORDLEN * NUMWORDS)

#ifndef STRINGSIZE
#define STRINGSIZE              1024
#endif /* STRINGSIZE */

#define STRCMP(a,b)		((a)[0]!=(b)[0]?1:strcmp((a),(b)))
#define CRACK_TOLOWER(a)	(isupper(a) ? tolower(a) : (a))
#define CRACK_TOUPPER(a)	(islower(a) ? toupper(a) : (a))

typedef unsigned char int8;
typedef unsigned short int int16;
typedef unsigned long int int32;

extern char dawgmagic[];

extern char **SplitOn();
extern char *Clone();
extern char *Mangle();
extern char *Trim();
extern char Chop();
extern char ChopNL();
extern int Debug();
extern int PackDAWG();
extern int ResetDAWG();
extern int UnPackDAWG();

/* ------------------------------------------------------------------ */

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
#define PFOR_WRITE			0x0001
#define PFOR_FLUSH			0x0002
#define PFOR_USEHWMS			0x0004
#define PFOR_SHAME			0x0008
#define PFOR_THESAKEOFTHECHILDREN	0x0010
#define PFOR_T5				0x0020
#define PFOR_APENNY			0x0040 /* pcl@ox.ac.uk */
    int32 hwms[256];
    struct pi_header header;
    int count;
    char data[NUMWORDS][MAXWORDLEN];	/* static arrays - ick! */
}
PWDICT;

extern PWDICT *PWOpen();
extern char *Mangle();
extern char *FascistCheck();

#define PW_WORDS(x) ((x)->header.pih_numwords)
#define PIH_MAGIC 0x70775635	/* pwV5 - will change */

/* ------------------------------------------------------------------ */
