/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 13, 2025.
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
#include <sys/stat.h>
#include <fcntl.h>
#include <ctype.h>
#include <string.h>
#include <time.h>

/* Utility library. */

#include "msg.h"
#include "htable.h"
#include "mymalloc.h"
#include "vstream.h"
#include "vstring.h"
#include "readlline.h"
#include "mac_expand.h"
#include "stringops.h"
#include "iostuff.h"
#include "name_mask.h"
#include "dict.h"
#include "dict_ht.h"
#include "warn_stat.h"
#include "line_number.h"

static HTABLE *dict_table;

 /*
  * Each (name, dictionary) instance has a reference count. The count is part
  * of the name, not the dictionary. The same dictionary may be registered
  * under multiple names. The structure below keeps track of instances and
  * reference counts.
  */
typedef struct {
    DICT   *dict;
    int     refcount;
} DICT_NODE;

#define dict_node(dict) \
	(dict_table ? (DICT_NODE *) htable_find(dict_table, dict) : 0)

/* Find a dictionary handle by name for lookup purposes. */

#define DICT_FIND_FOR_LOOKUP(dict, dict_name) do { \
    DICT_NODE *node; \
    if ((node = dict_node(dict_name)) != 0) \
	dict = node->dict; \
    else \
	dict = 0; \
} while (0)

/* Find a dictionary handle by name for update purposes. */

#define DICT_FIND_FOR_UPDATE(dict, dict_name) do { \
    DICT_NODE *node; \
    if ((node = dict_node(dict_name)) == 0) { \
	dict = dict_ht_open(dict_name, O_CREAT | O_RDWR, 0); \
	dict_register(dict_name, dict); \
    } else \
	dict = node->dict; \
} while (0)

#define STR(x)	vstring_str(x)

/* dict_register - make association with dictionary */

void    dict_register(const char *dict_name, DICT *dict_info)
{
    const char *myname = "dict_register";
    DICT_NODE *node;

    if (dict_table == 0)
	dict_table = htable_create(0);
    if ((node = dict_node(dict_name)) == 0) {
	node = (DICT_NODE *) mymalloc(sizeof(*node));
	node->dict = dict_info;
	node->refcount = 0;
	htable_enter(dict_table, dict_name, (void *) node);
    } else if (dict_info != node->dict)
	msg_fatal("%s: dictionary name exists: %s", myname, dict_name);
    node->refcount++;
    if (msg_verbose > 1)
	msg_info("%s: %s %d", myname, dict_name, node->refcount);
}

/* dict_handle - locate generic dictionary handle */

DICT   *dict_handle(const char *dict_name)
{
    DICT_NODE *node;

    return ((node = dict_node(dict_name)) != 0 ? node->dict : 0);
}

/* dict_node_free - dict_unregister() callback */

static void dict_node_free(void *ptr)
{
    DICT_NODE *node = (DICT_NODE *) ptr;
    DICT   *dict = node->dict;

    if (dict->close)
	dict->close(dict);
    myfree((void *) node);
}

/* dict_unregister - break association with named dictionary */

void    dict_unregister(const char *dict_name)
{
    const char *myname = "dict_unregister";
    DICT_NODE *node;

    if ((node = dict_node(dict_name)) == 0)
	msg_panic("non-existing dictionary: %s", dict_name);
    if (msg_verbose > 1)
	msg_info("%s: %s %d", myname, dict_name, node->refcount);
    if (--(node->refcount) == 0)
	htable_delete(dict_table, dict_name, dict_node_free);
}

/* dict_update - replace or add dictionary entry */

int     dict_update(const char *dict_name, const char *member, const char *value)
{
    const char *myname = "dict_update";
    DICT   *dict;

    DICT_FIND_FOR_UPDATE(dict, dict_name);
    if (msg_verbose > 1)
	msg_info("%s: %s = %s", myname, member, value);
    return (dict->update(dict, member, value));
}

/* dict_lookup - look up dictionary entry */

const char *dict_lookup(const char *dict_name, const char *member)
{
    const char *myname = "dict_lookup";
    DICT   *dict;
    const char *ret;

    DICT_FIND_FOR_LOOKUP(dict, dict_name);
    if (dict != 0) {
	ret = dict->lookup(dict, member);
	if (msg_verbose > 1)
	    msg_info("%s: %s = %s", myname, member, ret ? ret :
		     dict->error ? "(error)" : "(notfound)");
	return (ret);
    } else {
	if (msg_verbose > 1)
	    msg_info("%s: %s = %s", myname, member, "(notfound)");
	return (0);
    }
}

/* dict_delete - delete dictionary entry */

int     dict_delete(const char *dict_name, const char *member)
{
    const char *myname = "dict_delete";
    DICT   *dict;

    DICT_FIND_FOR_LOOKUP(dict, dict_name);
    if (msg_verbose > 1)
	msg_info("%s: delete %s", myname, member);
    return (dict ? dict->delete(dict, member) : DICT_STAT_FAIL);
}

/* dict_sequence - traverse dictionary */

int     dict_sequence(const char *dict_name, const int func,
		              const char **member, const char **value)
{
    const char *myname = "dict_sequence";
    DICT   *dict;

    DICT_FIND_FOR_LOOKUP(dict, dict_name);
    if (msg_verbose > 1)
	msg_info("%s: sequence func %d", myname, func);
    return (dict ? dict->sequence(dict, func, member, value) : DICT_STAT_FAIL);
}

/* dict_error - return last error */

int     dict_error(const char *dict_name)
{
    DICT   *dict;

    DICT_FIND_FOR_LOOKUP(dict, dict_name);
    return (dict ? dict->error : DICT_ERR_NONE);
}

/* dict_load_file_xt - read entries from text file */

int     dict_load_file_xt(const char *dict_name, const char *path)
{
    VSTREAM *fp;
    struct stat st;
    time_t  before;
    time_t  after;

    /*
     * Read the file again if it is hot. This may result in reading a partial
     * parameter name when a file changes in the middle of a read.
     */
    for (before = time((time_t *) 0); /* see below */ ; before = after) {
	if ((fp = vstream_fopen(path, O_RDONLY, 0)) == 0)
	    return (0);
	dict_load_fp(dict_name, fp);
	if (fstat(vstream_fileno(fp), &st) < 0)
	    msg_fatal("fstat %s: %m", path);
	if (vstream_ferror(fp) || vstream_fclose(fp))
	    msg_fatal("read %s: %m", path);
	after = time((time_t *) 0);
	if (st.st_mtime < before - 1 || st.st_mtime > after)
	    break;
	if (msg_verbose > 1)
	    msg_info("pausing to let %s cool down", path);
	doze(300000);
    }
    return (1);
}

/* dict_load_fp - read entries from open stream */

void    dict_load_fp(const char *dict_name, VSTREAM *fp)
{
    const char *myname = "dict_load_fp";
    VSTRING *buf;
    char   *member;
    char   *val;
    const char *old;
    int     last_line;
    int     lineno;
    const char *err;
    struct stat st;
    DICT   *dict;

    /*
     * Instantiate the dictionary even if the file is empty.
     */
    DICT_FIND_FOR_UPDATE(dict, dict_name);
    buf = vstring_alloc(100);
    last_line = 0;

    if (fstat(vstream_fileno(fp), &st) < 0)
	msg_fatal("fstat %s: %m", VSTREAM_PATH(fp));
    while (readllines(buf, fp, &last_line, &lineno)) {
	if ((err = split_nameval(STR(buf), &member, &val)) != 0)
	    msg_fatal("%s, line %d: %s: \"%s\"",
		      VSTREAM_PATH(fp),
		      lineno,
		      err, STR(buf));
	if (msg_verbose > 1)
	    msg_info("%s: %s = %s", myname, member, val);
	if ((old = dict->lookup(dict, member)) != 0
	    && strcmp(old, val) != 0)
	    msg_warn("%s, line %d: overriding earlier entry: %s=%s",
		     VSTREAM_PATH(fp), lineno, member, old);
	if (dict->update(dict, member, val) != 0)
	    msg_fatal("%s, line %d: unable to update %s:%s",
		      VSTREAM_PATH(fp), lineno, dict->type, dict->name);
    }
    vstring_free(buf);
    dict->owner.uid = st.st_uid;
    dict->owner.status = (st.st_uid != 0);
}

/* dict_eval_lookup - macro parser call-back routine */

static const char *dict_eval_lookup(const char *key, int unused_type,
				            void *context)
{
    char   *dict_name = (char *) context;
    const char *pp = 0;
    DICT   *dict;

    /*
     * XXX how would one recover?
     */
    DICT_FIND_FOR_LOOKUP(dict, dict_name);
    if (dict != 0
	&& (pp = dict->lookup(dict, key)) == 0 && dict->error != 0)
	msg_fatal("dictionary %s: lookup %s: operation failed", dict_name, key);
    return (pp);
}

/* dict_eval - expand embedded dictionary references */

const char *dict_eval(const char *dict_name, const char *value, int recursive)
{
    const char *myname = "dict_eval";
    static VSTRING *buf;
    int     status;

    /*
     * Initialize.
     */
    if (buf == 0)
	buf = vstring_alloc(10);

    /*
     * Expand macros, possibly recursively.
     */
#define DONT_FILTER (char *) 0

    status = mac_expand(buf, value,
			recursive ? MAC_EXP_FLAG_RECURSE : MAC_EXP_FLAG_NONE,
			DONT_FILTER, dict_eval_lookup, (void *) dict_name);
    if (status & MAC_PARSE_ERROR)
	msg_fatal("dictionary %s: macro processing error", dict_name);
    if (msg_verbose > 1) {
	if (strcmp(value, STR(buf)) != 0)
	    msg_info("%s: expand %s -> %s", myname, value, STR(buf));
	else
	    msg_info("%s: const  %s", myname, value);
    }
    return (STR(buf));
}

/* dict_walk - iterate over all dictionaries in arbitrary order */

void    dict_walk(DICT_WALK_ACTION action, void *ptr)
{
    HTABLE_INFO **ht_info_list;
    HTABLE_INFO **ht;
    HTABLE_INFO *h;

    ht_info_list = htable_list(dict_table);
    for (ht = ht_info_list; (h = *ht) != 0; ht++)
	action(h->key, (DICT *) h->value, ptr);
    myfree((void *) ht_info_list);
}

/* dict_changed_name - see if any dictionary has changed */

const char *dict_changed_name(void)
{
    const char *myname = "dict_changed_name";
    struct stat st;
    HTABLE_INFO **ht_info_list;
    HTABLE_INFO **ht;
    HTABLE_INFO *h;
    const char *status;
    DICT   *dict;

    ht_info_list = htable_list(dict_table);
    for (status = 0, ht = ht_info_list; status == 0 && (h = *ht) != 0; ht++) {
	dict = ((DICT_NODE *) h->value)->dict;
	if (dict->stat_fd < 0)			/* not file-based */
	    continue;
	if (dict->mtime == 0)			/* not bloody likely */
	    msg_warn("%s: table %s: null time stamp", myname, h->key);
	if (fstat(dict->stat_fd, &st) < 0)
	    msg_fatal("%s: fstat: %m", myname);
	if (((dict->flags & DICT_FLAG_MULTI_WRITER) == 0
	     && st.st_mtime != dict->mtime)
	    || st.st_nlink == 0)
	    status = h->key;
    }
    myfree((void *) ht_info_list);
    return (status);
}

/* dict_changed - backwards compatibility */

int     dict_changed(void)
{
    return (dict_changed_name() != 0);
}

 /*
  * Mapping between flag names and flag values.
  */
static const NAME_MASK dict_mask[] = {
    "warn_dup", DICT_FLAG_DUP_WARN,	/* if file, warn about dups */
    "ignore_dup", DICT_FLAG_DUP_IGNORE,	/* if file, ignore dups */
    "try0null", DICT_FLAG_TRY0NULL,	/* do not append 0 to key/value */
    "try1null", DICT_FLAG_TRY1NULL,	/* append 0 to key/value */
    "fixed", DICT_FLAG_FIXED,		/* fixed key map */
    "pattern", DICT_FLAG_PATTERN,	/* keys are patterns */
    "lock", DICT_FLAG_LOCK,		/* lock before access */
    "replace", DICT_FLAG_DUP_REPLACE,	/* if file, replace dups */
    "sync_update", DICT_FLAG_SYNC_UPDATE,	/* if file, sync updates */
    "debug", DICT_FLAG_DEBUG,		/* log access */
    "no_regsub", DICT_FLAG_NO_REGSUB,	/* disallow regexp substitution */
    "no_proxy", DICT_FLAG_NO_PROXY,	/* disallow proxy mapping */
    "no_unauth", DICT_FLAG_NO_UNAUTH,	/* disallow unauthenticated data */
    "fold_fix", DICT_FLAG_FOLD_FIX,	/* case-fold with fixed-case key map */
    "fold_mul", DICT_FLAG_FOLD_MUL,	/* case-fold with multi-case key map */
    "open_lock", DICT_FLAG_OPEN_LOCK,	/* permanent lock upon open */
    "bulk_update", DICT_FLAG_BULK_UPDATE,	/* bulk update if supported */
    "multi_writer", DICT_FLAG_MULTI_WRITER,	/* multi-writer safe */
    "utf8_request", DICT_FLAG_UTF8_REQUEST,	/* request UTF-8 activation */
    "utf8_active", DICT_FLAG_UTF8_ACTIVE,	/* UTF-8 is activated */
    0,
};

/* dict_flags_str - convert bitmask to symbolic flag names */

const char *dict_flags_str(int dict_flags)
{
    static VSTRING *buf = 0;

    if (buf == 0)
	buf = vstring_alloc(1);

    return (str_name_mask_opt(buf, "dictionary flags", dict_mask, dict_flags,
			      NAME_MASK_NUMBER | NAME_MASK_PIPE));
}

/* dict_flags_mask - convert symbolic flag names to bitmask */

int     dict_flags_mask(const char *names)
{
    return (name_mask("dictionary flags", dict_mask, names));
}
