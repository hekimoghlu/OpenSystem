/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 28, 2025.
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
/* $Id$ */

#ifndef __COMPILE_ET_H__
#define __COMPILE_ET_H__

#include <config.h>

#include <err.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <ctype.h>
#include <roken.h>

extern long base_id;
extern int number;
extern char *prefix;
extern char name[128];
extern char *id_str;
extern char *filename;
extern int numerror;

struct error_code {
    unsigned number;
    char *name;
    char *string;
    struct error_code *next, **tail;
};

extern struct error_code *codes;

#define APPEND(L, V) 				\
do {						\
    if((L) == NULL) {				\
	(L) = (V);				\
	(L)->tail = &(V)->next;			\
	(L)->next = NULL;			\
    }else{					\
	*(L)->tail = (V);			\
	(L)->tail = &(V)->next;			\
    }						\
}while(0)

#endif /* __COMPILE_ET_H__ */
