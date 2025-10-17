/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 16, 2022.
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
 * This is an open source non-commercial project. Dear PVS-Studio, please check it.
 * PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
 */

#include <config.h>

#include <stdlib.h>
#include <string.h>

#include "sudo_compat.h"
#include "sudo_queue.h"
#include "sudo_util.h"
#include "sudoers_debug.h"
#include "strlist.h"

struct sudoers_string *
sudoers_string_alloc(const char *s)
{
    struct sudoers_string *cs;
    debug_decl(sudoers_string_alloc, SUDOERS_DEBUG_UTIL);

    if ((cs = malloc(sizeof(*cs))) != NULL) {
	if ((cs->str = strdup(s)) == NULL) {
	    free(cs);
	    cs = NULL;
	}
    }

    debug_return_ptr(cs);
}

void
sudoers_string_free(struct sudoers_string *cs)
{
    if (cs != NULL) {
	free(cs->str);
	free(cs);
    }
}

struct sudoers_str_list *
str_list_alloc(void)
{
    struct sudoers_str_list *strlist;
    debug_decl(str_list_alloc, SUDOERS_DEBUG_UTIL);

    strlist = malloc(sizeof(*strlist));
    if (strlist != NULL) {
	STAILQ_INIT(strlist);
	strlist->refcnt = 1;
    }

    debug_return_ptr(strlist);
}

void
str_list_free(void *v)
{
    struct sudoers_str_list *strlist = v;
    struct sudoers_string *first;
    debug_decl(str_list_free, SUDOERS_DEBUG_UTIL);

    if (strlist != NULL) {
	if (--strlist->refcnt == 0) {
	    while ((first = STAILQ_FIRST(strlist)) != NULL) {
		STAILQ_REMOVE_HEAD(strlist, entries);
		sudoers_string_free(first);
	    }
	    free(strlist);
	}
    }
    debug_return;
}
