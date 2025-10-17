/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 17, 2021.
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
/*
**
**  NAME
**
**      GETFLAGS.H
**
**  FACILITY:
**
**      Interface Definition Language (IDL) Compiler
**
**  ABSTRACT:
**
**      Constants for command line parsing
**
**  VERSION: DCE 1.0
**
*/

#ifndef GETFLAGS_H
#define GETFLAGS_H

#include <nidl.h>

typedef char *FLAGDEST;
typedef struct options
        {
        const char *option;
        int ftype;
        FLAGDEST dest;
        } OPTIONS;

/*
 * Rico, 23-Mar-90: The format of the ftype field appears to be:
 *
 *      Bit 15:   (1 bit)  HIDARG flag - flags hidden args to printflags fn
 *      Bit 14:   (1 bit)  VARARG flag - flags a variable option (can repeat)
 *      Bit 13-8: (6 bits) maximum number of occurences of a variable option
 *      Bit  7-0: (8 bits) argument type
 *
 *       15  14  13              8   7                    0
 *      -----------------------------------------------------
 *      | H | V |  max_occurences  |     argument_type      |
 *      -----------------------------------------------------
 */

/* Argument types */
#define INTARG      0
#define STRARG      1
#define TOGGLEARG   2
#define CHRARG      3
#define FLTARG      4
#define LONGARG     5
#define ASSERTARG   6
#define DENYARG     7
#define OSTRARG     8           /* Optional string arg, added 23-Mar-90 */

#define HIDARG (128 << 8)       /* H bit */
#define VARARGFLAG 64           /* V bit - gets shifted 8 bits by macros */
#define MULTARGMASK 63          /* Mask to get max_occurences */

/* Macros for specifying ftype */
#define MULTARG(n, a) (((n) << 8) + a)
#define AINTARG(n) MULTARG(n,INTARG)
#define VINTARG(n) AINTARG(n|VARARGFLAG)
#define ASTRARG(n) MULTARG(n,STRARG)
#define VSTRARG(n) ASTRARG(n|VARARGFLAG)
#define ATOGGLEARG(n) MULTARG(n,TOGGLEARG)
#define AASSERTARG(n) MULTARG(n,ASSERTARG)
#define ADENYARG(n) MULTARG(n,DENYARG)
#define ACHRARG(n) MULTARG(n,CHRARG)
#define VCHRARG(n) ACHRARG(n|VARARGFLAG)
#define AFLTARG(n) MULTARG(n,FLTARG)
#define VFLTARG(n) AFLTARG(n|VARARGFLAG)
#define ALONGARG(n) MULTARG(n,LONGARG)
#define VLONGARG(n) AFLTARG(n|VARARGFLAG)

/* Macros for converting command line arguments */
#define GETINT(s) {int __temp__ = atoi(*++av); memcpy(s, &__temp__, sizeof(int)); ac--;}
#define GETSTR(s) {memcpy(s, ++av, sizeof(char *)); ac--;}
#define GETCH(s) {av++; s = av[0][0]; ac--;}
#define GETFLT(s) {double __temp__ = atof(*++av); memcpy(s, &__temp__, sizeof(double)); ac--;}
#define GETLONG(s) {long __temp__ = atol(*++av); memcpy(s, &__temp__, sizeof(long)); ac--;}

void printflags (
    const OPTIONS table[]
);

void getflags (
    int argc,
    char **argv,
    const OPTIONS table[]
);

void flags_incr_count (
    const OPTIONS table[],
    const char *option,
    int delta
);

int flags_option_count (
    const OPTIONS table[],
    const char *option
);

int flags_other_count (
    void
);

char *flags_other (
    int index
);

#endif
