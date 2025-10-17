/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 8, 2023.
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
#include <errno.h>

/* Utility library. */

#include <mymalloc.h>
#include <msg.h>
#include <compat_va_copy.h>
#include <dict.h>

/* Application-specific. */

typedef struct {
    DICT    dict;			/* generic members */
    char   *reason;			/* open failure reason */
} DICT_SURROGATE;

/* dict_surrogate_sequence - fail lookup */

static int dict_surrogate_sequence(DICT *dict, int unused_func,
			               const char **key, const char **value)
{
    DICT_SURROGATE *dp = (DICT_SURROGATE *) dict;

    msg_warn("%s:%s is unavailable. %s",
	     dict->type, dict->name, dp->reason);
    DICT_ERR_VAL_RETURN(dict, DICT_ERR_RETRY, DICT_STAT_ERROR);
}

/* dict_surrogate_update - fail lookup */

static int dict_surrogate_update(DICT *dict, const char *unused_name,
				         const char *unused_value)
{
    DICT_SURROGATE *dp = (DICT_SURROGATE *) dict;

    msg_warn("%s:%s is unavailable. %s",
	     dict->type, dict->name, dp->reason);
    DICT_ERR_VAL_RETURN(dict, DICT_ERR_RETRY, DICT_STAT_ERROR);
}

/* dict_surrogate_lookup - fail lookup */

static const char *dict_surrogate_lookup(DICT *dict, const char *unused_name)
{
    DICT_SURROGATE *dp = (DICT_SURROGATE *) dict;

    msg_warn("%s:%s is unavailable. %s",
	     dict->type, dict->name, dp->reason);
    DICT_ERR_VAL_RETURN(dict, DICT_ERR_RETRY, (char *) 0);
}

/* dict_surrogate_delete - fail delete */

static int dict_surrogate_delete(DICT *dict, const char *unused_name)
{
    DICT_SURROGATE *dp = (DICT_SURROGATE *) dict;

    msg_warn("%s:%s is unavailable. %s",
	     dict->type, dict->name, dp->reason);
    DICT_ERR_VAL_RETURN(dict, DICT_ERR_RETRY, DICT_STAT_ERROR);
}

/* dict_surrogate_close - close fail dictionary */

static void dict_surrogate_close(DICT *dict)
{
    DICT_SURROGATE *dp = (DICT_SURROGATE *) dict;

    myfree((void *) dp->reason);
    dict_free(dict);
}

int     dict_allow_surrogate = 0;

/* dict_surrogate - terminate or provide surrogate dictionary */

DICT   *dict_surrogate(const char *dict_type, const char *dict_name,
		               int open_flags, int dict_flags,
		               const char *fmt,...)
{
    va_list ap;
    va_list ap2;
    DICT_SURROGATE *dp;
    VSTRING *buf;
    void    (*log_fn) (const char *, va_list);
    int     saved_errno = errno;

    /*
     * Initialize argument lists.
     */
    va_start(ap, fmt);
    VA_COPY(ap2, ap);

    /*
     * Log the problem immediately when it is detected. The table may not be
     * accessed in every program execution (that is the whole point of
     * continuing with reduced functionality) but we don't want the problem
     * to remain unnoticed until long after a configuration mistake is made.
     */
    log_fn = dict_allow_surrogate ? vmsg_error : vmsg_fatal;
    log_fn(fmt, ap);
    va_end(ap);

    /*
     * Log the problem upon each access.
     */
    dp = (DICT_SURROGATE *) dict_alloc(dict_type, dict_name, sizeof(*dp));
    dp->dict.lookup = dict_surrogate_lookup;
    if (open_flags & O_RDWR) {
	dp->dict.update = dict_surrogate_update;
	dp->dict.delete = dict_surrogate_delete;
    }
    dp->dict.sequence = dict_surrogate_sequence;
    dp->dict.close = dict_surrogate_close;
    dp->dict.flags = dict_flags | DICT_FLAG_PATTERN;
    dp->dict.owner.status = DICT_OWNER_TRUSTED;
    buf = vstring_alloc(10);
    errno = saved_errno;
    vstring_vsprintf(buf, fmt, ap2);
    va_end(ap2);
    dp->reason = vstring_export(buf);
    return (DICT_DEBUG (&dp->dict));
}
