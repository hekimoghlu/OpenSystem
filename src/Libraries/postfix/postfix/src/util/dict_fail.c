/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 25, 2021.
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

/* Utility library. */

#include <mymalloc.h>
#include <msg.h>
#include <dict.h>
#include <dict_fail.h>

/* Application-specific. */

typedef struct {
    DICT    dict;			/* generic members */
    int     dict_errno;			/* fixed error result */
} DICT_FAIL;

/* dict_fail_sequence - fail lookup */

static int dict_fail_sequence(DICT *dict, int unused_func,
			              const char **key, const char **value)
{
    DICT_FAIL *dp = (DICT_FAIL *) dict;

    DICT_ERR_VAL_RETURN(dict, dp->dict_errno, DICT_STAT_ERROR);
}

/* dict_fail_update - fail lookup */

static int dict_fail_update(DICT *dict, const char *unused_name,
			            const char *unused_value)
{
    DICT_FAIL *dp = (DICT_FAIL *) dict;

    DICT_ERR_VAL_RETURN(dict, dp->dict_errno, DICT_STAT_ERROR);
}

/* dict_fail_lookup - fail lookup */

static const char *dict_fail_lookup(DICT *dict, const char *unused_name)
{
    DICT_FAIL *dp = (DICT_FAIL *) dict;

    DICT_ERR_VAL_RETURN(dict, dp->dict_errno, (char *) 0);
}

/* dict_fail_delete - fail delete */

static int dict_fail_delete(DICT *dict, const char *unused_name)
{
    DICT_FAIL *dp = (DICT_FAIL *) dict;

    DICT_ERR_VAL_RETURN(dict, dp->dict_errno, DICT_STAT_ERROR);
}

/* dict_fail_close - close fail dictionary */

static void dict_fail_close(DICT *dict)
{
    dict_free(dict);
}

/* dict_fail_open - make association with fail variable */

DICT   *dict_fail_open(const char *name, int open_flags, int dict_flags)
{
    DICT_FAIL *dp;

    dp = (DICT_FAIL *) dict_alloc(DICT_TYPE_FAIL, name, sizeof(*dp));
    dp->dict.lookup = dict_fail_lookup;
    if (open_flags & O_RDWR) {
	dp->dict.update = dict_fail_update;
	dp->dict.delete = dict_fail_delete;
    }
    dp->dict.sequence = dict_fail_sequence;
    dp->dict.close = dict_fail_close;
    dp->dict.flags = dict_flags | DICT_FLAG_PATTERN;
    dp->dict_errno = DICT_ERR_RETRY;
    dp->dict.owner.status = DICT_OWNER_TRUSTED;
    return (DICT_DEBUG (&dp->dict));
}
