/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 19, 2024.
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
/* System libraries. */

#include "sys_defs.h"

/* Utility library. */

#include "msg.h"
#include "mymalloc.h"
#include "myflock.h"
#include "dict.h"

/* dict_default_lookup - trap unimplemented operation */

static const char *dict_default_lookup(DICT *dict, const char *unused_key)
{
    msg_fatal("table %s:%s: lookup operation is not supported",
	      dict->type, dict->name);
}

/* dict_default_update - trap unimplemented operation */

static int dict_default_update(DICT *dict, const char *unused_key,
			               const char *unused_value)
{
    msg_fatal("table %s:%s: update operation is not supported",
	      dict->type, dict->name);
}

/* dict_default_delete - trap unimplemented operation */

static int dict_default_delete(DICT *dict, const char *unused_key)
{
    msg_fatal("table %s:%s: delete operation is not supported",
	      dict->type, dict->name);
}

/* dict_default_sequence - trap unimplemented operation */

static int dict_default_sequence(DICT *dict, int unused_function,
		         const char **unused_key, const char **unused_value)
{
    msg_fatal("table %s:%s: sequence operation is not supported",
	      dict->type, dict->name);
}

/* dict_default_lock - default lock handler */

static int dict_default_lock(DICT *dict, int operation)
{
    if (dict->lock_fd >= 0) {
	return (myflock(dict->lock_fd, dict->lock_type, operation));
    } else {
	return (0);
    }
}

/* dict_default_close - trap unimplemented operation */

static void dict_default_close(DICT *dict)
{
    msg_fatal("table %s:%s: close operation is not supported",
	      dict->type, dict->name);
}

/* dict_alloc - allocate dictionary object, initialize super-class */

DICT   *dict_alloc(const char *dict_type, const char *dict_name, ssize_t size)
{
    DICT   *dict = (DICT *) mymalloc(size);

    dict->type = mystrdup(dict_type);
    dict->name = mystrdup(dict_name);
    dict->flags = DICT_FLAG_FIXED;
    dict->lookup = dict_default_lookup;
    dict->update = dict_default_update;
    dict->delete = dict_default_delete;
    dict->sequence = dict_default_sequence;
    dict->close = dict_default_close;
    dict->lock = dict_default_lock;
    dict->lock_type = INTERNAL_LOCK;
    dict->lock_fd = -1;
    dict->stat_fd = -1;
    dict->mtime = 0;
    dict->fold_buf = 0;
    dict->owner.status = DICT_OWNER_UNKNOWN;
    dict->owner.uid = INT_MAX;
    dict->error = DICT_ERR_NONE;
    dict->jbuf = 0;
    dict->utf8_backup = 0;
    return dict;
}

/* dict_free - super-class destructor */

void    dict_free(DICT *dict)
{
    myfree(dict->type);
    myfree(dict->name);
    if (dict->jbuf)
	myfree((void *) dict->jbuf);
    if (dict->utf8_backup)
	myfree((void *) dict->utf8_backup);
    myfree((void *) dict);
}

 /*
  * TODO: add a dict_flags() argument to dict_alloc() and handle jump buffer
  * allocation there.
  */

/* dict_jmp_alloc - enable exception handling */

void    dict_jmp_alloc(DICT *dict)
{
    if (dict->jbuf == 0)
	dict->jbuf = (DICT_JMP_BUF *) mymalloc(sizeof(DICT_JMP_BUF));
}
