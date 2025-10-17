/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 21, 2023.
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

#ifndef _NAME_CODE_H_INCLUDED_
#define _NAME_CODE_H_INCLUDED_

/*++
/* NAME
/*	name_mask 3h
/* SUMMARY
/*	name to number table mapping
/* SYNOPSIS
/*	#include <name_code.h>
/* DESCRIPTION
/* .nf

 /*
  * External interface.
  */
typedef struct {
    const char *name;
    int     code;
} NAME_CODE;

#define NAME_CODE_FLAG_NONE		0
#define NAME_CODE_FLAG_STRICT_CASE	(1<<0)

extern int name_code(const NAME_CODE *, int, const char *);
extern const char *str_name_code(const NAME_CODE *, int);

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
