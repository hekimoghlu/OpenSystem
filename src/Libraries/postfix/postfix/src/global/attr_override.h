/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 17, 2023.
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

#ifndef _ATTR_OVERRIDE_H_INCLUDED_
#define _ATTR_OVERRIDE_H_INCLUDED_

/*++
/* NAME
/*	attr_override 3h
/* SUMMARY
/*	apply name=value settings from string
/* SYNOPSIS
/*	#include <attr_override.h>
/* DESCRIPTION
/* .nf

 /*
  * External interface.
  */
#include <check_arg.h>

extern void attr_override(char *, const char *, const char *,...);

typedef struct {
    const char *name;
    CONST_CHAR_STAR *target;
    int     min;
    int     max;
} ATTR_OVER_STR;

typedef struct {
    const char *name;
    const char *defval;
    int    *target;
    int     min;
    int     max;
} ATTR_OVER_TIME;

typedef struct {
    const char *name;
    int    *target;
    int     min;
    int     max;
} ATTR_OVER_INT;

/* Type-unchecked API, internal use only. */
#define ATTR_OVER_END		0
#define ATTR_OVER_STR_TABLE	1
#define ATTR_OVER_TIME_TABLE	2
#define ATTR_OVER_INT_TABLE	3

/* Type-checked API, external use only. */
#define CA_ATTR_OVER_END		0
#define CA_ATTR_OVER_STR_TABLE(v)	ATTR_OVER_STR_TABLE, CHECK_CPTR(ATTR_OVER, ATTR_OVER_STR, (v))
#define CA_ATTR_OVER_TIME_TABLE(v)	ATTR_OVER_TIME_TABLE, CHECK_CPTR(ATTR_OVER, ATTR_OVER_TIME, (v))
#define CA_ATTR_OVER_INT_TABLE(v)	ATTR_OVER_INT_TABLE, CHECK_CPTR(ATTR_OVER, ATTR_OVER_INT, (v))

CHECK_CPTR_HELPER_DCL(ATTR_OVER, ATTR_OVER_TIME);
CHECK_CPTR_HELPER_DCL(ATTR_OVER, ATTR_OVER_STR);
CHECK_CPTR_HELPER_DCL(ATTR_OVER, ATTR_OVER_INT);

/* LICENSE
/* .ad
/* .fi
/*	The Secure Mailer license must be distributed with this software.
/* AUTHOR(S)
/*	Wietse Venema
/*	IBM T.J. Watson Research
/*	P.O. Box 704
/*	Yorktown Heights, NY 10598, USA
/*--*/

#endif
