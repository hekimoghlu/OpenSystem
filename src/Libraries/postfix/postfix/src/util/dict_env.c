/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 7, 2024.
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
#include "safe.h"
#include "stringops.h"
#include "dict.h"
#include "dict_env.h"

/* dict_env_update - update environment array */

static int dict_env_update(DICT *dict, const char *name, const char *value)
{
    dict->error = 0;

    /*
     * Optionally fold the key.
     */
    if (dict->flags & DICT_FLAG_FOLD_FIX) {
	if (dict->fold_buf == 0)
	    dict->fold_buf = vstring_alloc(10);
	vstring_strcpy(dict->fold_buf, name);
	name = lowercase(vstring_str(dict->fold_buf));
    }
    if (setenv(name, value, 1))
	msg_fatal("setenv: %m");

    return (DICT_STAT_SUCCESS);
}

/* dict_env_lookup - access environment array */

static const char *dict_env_lookup(DICT *dict, const char *name)
{
    dict->error = 0;

    /*
     * Optionally fold the key.
     */
    if (dict->flags & DICT_FLAG_FOLD_FIX) {
	if (dict->fold_buf == 0)
	    dict->fold_buf = vstring_alloc(10);
	vstring_strcpy(dict->fold_buf, name);
	name = lowercase(vstring_str(dict->fold_buf));
    }
    return (safe_getenv(name));
}

/* dict_env_close - close environment dictionary */

static void dict_env_close(DICT *dict)
{
    if (dict->fold_buf)
	vstring_free(dict->fold_buf);
    dict_free(dict);
}

/* dict_env_open - make association with environment array */

DICT   *dict_env_open(const char *name, int unused_flags, int dict_flags)
{
    DICT   *dict;

    dict = dict_alloc(DICT_TYPE_ENVIRON, name, sizeof(*dict));
    dict->lookup = dict_env_lookup;
    dict->update = dict_env_update;
    dict->close = dict_env_close;
    dict->flags = dict_flags | DICT_FLAG_FIXED;
    if (dict_flags & DICT_FLAG_FOLD_FIX)
	dict->fold_buf = vstring_alloc(10);
    dict->owner.status = DICT_OWNER_TRUSTED;
    return (DICT_DEBUG (dict));
}
