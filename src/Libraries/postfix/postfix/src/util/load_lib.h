/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 21, 2023.
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

#ifndef _LOAD_LIB_H_INCLUDED_
#define _LOAD_LIB_H_INCLUDED_

/*++
/* NAME
/*	load_lib 3h
/* SUMMARY
/*	library loading wrappers
/* SYNOPSIS
/*	#include "load_lib.h"
/* DESCRIPTION
/* .nf

 /*
  * External interface.
  */
/* NULL name terminates list */
typedef struct LIB_FN {
    const char *name;
    void  (*fptr)(void);
} LIB_FN;

typedef struct LIB_DP {
    const char *name;
    void  *dptr;
} LIB_DP;

extern void load_library_symbols(const char *, LIB_FN *, LIB_DP *);

/* LICENSE
/* .ad
/* .fi
/*	The Secure Mailer license must be distributed with this software.
/* AUTHOR(S)
/*	LaMont Jones
/*	Hewlett-Packard Company
/*	3404 Harmony Road
/*	Fort Collins, CO 80528, USA
/*
/*	Wietse Venema
/*	IBM T.J. Watson Research
/*	P.O. Box 704
/*	Yorktown Heights, NY 10598, USA
/*--*/

#endif
