/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 12, 2024.
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
#undef compctlread

typedef struct compctlp  *Compctlp;
typedef struct compctl   *Compctl;
typedef struct compcond  *Compcond;
typedef struct patcomp   *Patcomp;

/* node for compctl hash table (compctltab) */

struct compctlp {
    struct hashnode node;
    Compctl cc;			/* pointer to the compctl desc.     */
};

/* for the list of pattern compctls */

struct patcomp {
    Patcomp next;
    char *pat;
    Compctl cc;
};

/* compctl -x condition */

struct compcond {
    Compcond and, or;		/* the next or'ed/and'ed conditions    */
    int type;			/* the type (CCT_*)                    */
    int n;			/* the array length                    */
    union {			/* these structs hold the data used to */
	struct {		/* test this condition                 */
	    int *a, *b;		/* CCT_POS, CCT_NUMWORDS               */
	}
	r;
	struct {		/* CCT_CURSTR, CCT_CURPAT,... */
	    int *p;
	    char **s;
	}
	s;
	struct {		/* CCT_RANGESTR,... */
	    char **a, **b;
	}
	l;
    }
    u;
};

#define CCT_UNUSED     0
#define CCT_POS        1
#define CCT_CURSTR     2
#define CCT_CURPAT     3
#define CCT_WORDSTR    4
#define CCT_WORDPAT    5
#define CCT_CURSUF     6
#define CCT_CURPRE     7
#define CCT_CURSUB     8
#define CCT_CURSUBC    9
#define CCT_NUMWORDS  10
#define CCT_RANGESTR  11
#define CCT_RANGEPAT  12
#define CCT_QUOTE     13

/* Contains the real description for compctls */

struct compctl {
    int refc;			/* reference count                         */
    Compctl next;		/* next compctl for -x                     */
    unsigned long mask, mask2;	/* masks of things to complete (CC_*)      */
    char *keyvar;		/* for -k (variable)                       */
    char *glob;			/* for -g (globbing)                       */
    char *str;			/* for -s (expansion)                      */
    char *func;			/* for -K (function)                       */
    char *explain;		/* for -X (explanation)                    */
    char *ylist;		/* for -y (user-defined desc. for listing) */
    char *prefix, *suffix;	/* for -P and -S (prefix, suffix)          */
    char *subcmd;		/* for -l (command name to use)            */
    char *substr;		/* for -1 (command name to use)            */
    char *withd;		/* for -w (with directory                  */
    char *hpat;			/* for -H (history pattern)                */
    int hnum;			/* for -H (number of events to search)     */
    char *gname;		/* for -J and -V (group name)              */
    Compctl ext;		/* for -x (first of the compctls after -x) */
    Compcond cond;		/* for -x (condition for this compctl)     */
    Compctl xor;		/* for + (next of the xor'ed compctls)     */
    Cmatcher matcher;		/* matcher control (-M) */
    char *mstr;			/* matcher string */
};

/* objects to complete (mask) */
#define CC_FILES	(1<<0)
#define CC_COMMPATH	(1<<1)
#define CC_REMOVE	(1<<2)
#define CC_OPTIONS	(1<<3)
#define CC_VARS		(1<<4)
#define CC_BINDINGS	(1<<5)
#define CC_ARRAYS	(1<<6)
#define CC_INTVARS	(1<<7)
#define CC_SHFUNCS	(1<<8)
#define CC_PARAMS	(1<<9)
#define CC_ENVVARS	(1<<10)
#define CC_JOBS		(1<<11)
#define CC_RUNNING	(1<<12)
#define CC_STOPPED	(1<<13)
#define CC_BUILTINS	(1<<14)
#define CC_ALREG	(1<<15)
#define CC_ALGLOB	(1<<16)
#define CC_USERS	(1<<17)
#define CC_DISCMDS	(1<<18)
#define CC_EXCMDS	(1<<19)
#define CC_SCALARS	(1<<20)
#define CC_READONLYS	(1<<21)
#define CC_SPECIALS	(1<<22)
#define CC_DELETE	(1<<23)
#define CC_NAMED	(1<<24)
#define CC_QUOTEFLAG	(1<<25)
#define CC_EXTCMDS	(1<<26)
#define CC_RESWDS	(1<<27)
#define CC_DIRS		(1<<28)

#define CC_EXPANDEXPL	(1<<30)
#define CC_RESERVED	(1<<31)

/* objects to complete (mask2) */
#define CC_NOSORT	(1<<0)
#define CC_XORCONT	(1<<1)
#define CC_CCCONT	(1<<2)
#define CC_PATCONT	(1<<3)
#define CC_DEFCONT	(1<<4)
#define CC_UNIQCON      (1<<5)
#define CC_UNIQALL      (1<<6)
