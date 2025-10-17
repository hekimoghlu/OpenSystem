/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 23, 2025.
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

#include "sys_defs.h"
#include <stdio.h>			/* sprintf() prototype */
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

/* Utility library. */

#include "mymalloc.h"
#include "msg.h"
#include "stringops.h"
#include "dict.h"
#include "dict_static.h"

/* dict_static_lookup - access static value*/

static const char *dict_static_lookup(DICT *dict, const char *unused_name)
{
    DICT_ERR_VAL_RETURN(dict, DICT_ERR_NONE, dict->name);
}

/* dict_static_close - close static dictionary */

static void dict_static_close(DICT *dict)
{
    dict_free(dict);
}

/* dict_static_open - make association with static variable */

DICT   *dict_static_open(const char *name, int open_flags, int dict_flags)
{
    DICT   *dict;
    char   *err = 0;
    char   *cp, *saved_name = 0;

    /*
     * Let the optimizer worry about eliminating redundant code.
     */
#define DICT_STATIC_OPEN_RETURN(d) do { \
        DICT *__d = (d); \
        if (saved_name != 0) \
            myfree(saved_name); \
        if (err != 0) \
            myfree(err); \
        return (__d); \
    } while (0)

    /*
     * Optionally strip surrounding braces and whitespace.
     */
    if (name[0] == CHARS_BRACE[0]) {
	saved_name = cp = mystrdup(name);
	if ((err = extpar(&cp, CHARS_BRACE, EXTPAR_FLAG_STRIP)) != 0)
	    DICT_STATIC_OPEN_RETURN(dict_surrogate(DICT_TYPE_STATIC, name,
						   open_flags, dict_flags,
						   "bad %s:name syntax: %s",
						   DICT_TYPE_STATIC, err));
	name = cp;
    }

    /*
     * Bundle up the request.
     */
    dict = dict_alloc(DICT_TYPE_STATIC, name, sizeof(*dict));
    dict->lookup = dict_static_lookup;
    dict->close = dict_static_close;
    dict->flags = dict_flags | DICT_FLAG_FIXED;
    dict->owner.status = DICT_OWNER_TRUSTED;
    DICT_STATIC_OPEN_RETURN(DICT_DEBUG (dict));
}
