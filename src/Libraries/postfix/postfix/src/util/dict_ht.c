/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 8, 2022.
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

/* Utility library. */

#include "mymalloc.h"
#include "htable.h"
#include "dict.h"
#include "dict_ht.h"
#include "stringops.h"
#include "vstring.h"

/* Application-specific. */

typedef struct {
    DICT    dict;			/* generic members */
    HTABLE *table;			/* hash table */
} DICT_HT;

/* dict_ht_delete - delete hash-table entry */

static int dict_ht_delete(DICT *dict, const char *name)
{
    DICT_HT *dict_ht = (DICT_HT *) dict;

    /*
     * Optionally fold the key.
     */
    if (dict->flags & DICT_FLAG_FOLD_FIX) {
	if (dict->fold_buf == 0)
	    dict->fold_buf = vstring_alloc(10);
	vstring_strcpy(dict->fold_buf, name);
	name = lowercase(vstring_str(dict->fold_buf));
    }
    if (htable_locate(dict_ht->table, name) == 0) {
	DICT_ERR_VAL_RETURN(dict, DICT_ERR_NONE, DICT_STAT_FAIL);
    } else {
	htable_delete(dict_ht->table, name, myfree);
	DICT_ERR_VAL_RETURN(dict, DICT_ERR_NONE, DICT_STAT_SUCCESS);
    }
}

/* dict_ht_lookup - find hash-table entry */

static const char *dict_ht_lookup(DICT *dict, const char *name)
{
    DICT_HT *dict_ht = (DICT_HT *) dict;

    /*
     * Optionally fold the key.
     */
    if (dict->flags & DICT_FLAG_FOLD_FIX) {
	if (dict->fold_buf == 0)
	    dict->fold_buf = vstring_alloc(10);
	vstring_strcpy(dict->fold_buf, name);
	name = lowercase(vstring_str(dict->fold_buf));
    }
    DICT_ERR_VAL_RETURN(dict, DICT_ERR_NONE, htable_find(dict_ht->table, name));
}

/* dict_ht_update - add or update hash-table entry */

static int dict_ht_update(DICT *dict, const char *name, const char *value)
{
    DICT_HT *dict_ht = (DICT_HT *) dict;
    HTABLE_INFO *ht;
    char   *saved_value = mystrdup(value);

    /*
     * Optionally fold the key.
     */
    if (dict->flags & DICT_FLAG_FOLD_FIX) {
	if (dict->fold_buf == 0)
	    dict->fold_buf = vstring_alloc(10);
	vstring_strcpy(dict->fold_buf, name);
	name = lowercase(vstring_str(dict->fold_buf));
    }
    if ((ht = htable_locate(dict_ht->table, name)) != 0) {
	myfree(ht->value);
    } else {
	ht = htable_enter(dict_ht->table, name, (void *) 0);
    }
    ht->value = saved_value;
    DICT_ERR_VAL_RETURN(dict, DICT_ERR_NONE, DICT_STAT_SUCCESS);
}

/* dict_ht_sequence - first/next iterator */

static int dict_ht_sequence(DICT *dict, int how, const char **name,
			            const char **value)
{
    DICT_HT *dict_ht = (DICT_HT *) dict;
    HTABLE_INFO *ht;

    ht = htable_sequence(dict_ht->table,
			 how == DICT_SEQ_FUN_FIRST ? HTABLE_SEQ_FIRST :
			 how == DICT_SEQ_FUN_NEXT ? HTABLE_SEQ_NEXT :
			 HTABLE_SEQ_STOP);
    if (ht != 0) {
	*name = ht->key;
	*value = ht->value;
	DICT_ERR_VAL_RETURN(dict, DICT_ERR_NONE, DICT_STAT_SUCCESS);
    } else {
	*name = 0;
	*value = 0;
	DICT_ERR_VAL_RETURN(dict, DICT_ERR_NONE, DICT_STAT_FAIL);
    }
}

/* dict_ht_close - disassociate from hash table */

static void dict_ht_close(DICT *dict)
{
    DICT_HT *dict_ht = (DICT_HT *) dict;

    htable_free(dict_ht->table, myfree);
    if (dict_ht->dict.fold_buf)
	vstring_free(dict_ht->dict.fold_buf);
    dict_free(dict);
}

/* dict_ht_open - create association with hash table */

DICT   *dict_ht_open(const char *name, int unused_open_flags, int dict_flags)
{
    DICT_HT *dict_ht;

    dict_ht = (DICT_HT *) dict_alloc(DICT_TYPE_HT, name, sizeof(*dict_ht));
    dict_ht->dict.lookup = dict_ht_lookup;
    dict_ht->dict.update = dict_ht_update;
    dict_ht->dict.delete = dict_ht_delete;
    dict_ht->dict.sequence = dict_ht_sequence;
    dict_ht->dict.close = dict_ht_close;
    dict_ht->dict.flags = dict_flags | DICT_FLAG_FIXED;
    if (dict_flags & DICT_FLAG_FOLD_FIX)
	dict_ht->dict.fold_buf = vstring_alloc(10);
    dict_ht->table = htable_create(0);
    dict_ht->dict.owner.status = DICT_OWNER_TRUSTED;
    return (&dict_ht->dict);
}
