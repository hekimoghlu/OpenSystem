/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 9, 2023.
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
/* System library. */

#include <sys_defs.h>
#include <string.h>

/* Utility library. */

#include <mymalloc.h>
#include <vstring.h>

/* Global library. */

#include "tok822.h"

/* tok822_alloc - allocate and initialize token */

TOK822 *tok822_alloc(int type, const char *strval)
{
    TOK822 *tp;

#define CONTAINER_TOKEN(x) \
	((x) == TOK822_ADDR || (x) == TOK822_STARTGRP)

    tp = (TOK822 *) mymalloc(sizeof(*tp));
    tp->type = type;
    tp->next = tp->prev = tp->head = tp->tail = tp->owner = 0;
    tp->vstr = (type < TOK822_MINTOK || CONTAINER_TOKEN(type) ? 0 :
		strval == 0 ? vstring_alloc(10) :
		vstring_strcpy(vstring_alloc(strlen(strval) + 1), strval));
    return (tp);
}

/* tok822_free - destroy token */

TOK822 *tok822_free(TOK822 *tp)
{
    if (tp->vstr)
	vstring_free(tp->vstr);
    myfree((void *) tp);
    return (0);
}
