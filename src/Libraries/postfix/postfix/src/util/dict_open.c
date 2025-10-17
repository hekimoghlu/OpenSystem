/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 4, 2024.
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
#include <stdlib.h>

/* Utility library. */

#include <argv.h>
#include <mymalloc.h>
#include <msg.h>
#include <dict.h>
#include <dict_cdb.h>
#include <dict_env.h>
#include <dict_unix.h>
#include <dict_tcp.h>
#include <dict_sdbm.h>
#include <dict_dbm.h>
#include <dict_db.h>
#include <dict_lmdb.h>
#include <dict_nis.h>
#include <dict_nisplus.h>
#include <dict_ni.h>
#include <dict_pcre.h>
#include <dict_regexp.h>
#include <dict_static.h>
#include <dict_cidr.h>
#include <dict_ht.h>
#include <dict_thash.h>
#include <dict_sockmap.h>
#include <dict_fail.h>
#include <dict_pipe.h>
#include <dict_random.h>
#include <dict_union.h>
#include <dict_inline.h>
#include <stringops.h>
#include <split_at.h>
#include <htable.h>
#include <myflock.h>

 /*
  * lookup table for available map types.
  */
typedef struct {
    char   *type;
    DICT_OPEN_FN open;
} DICT_OPEN_INFO;

static const DICT_OPEN_INFO dict_open_info[] = {
    DICT_TYPE_ENVIRON, dict_env_open,
    DICT_TYPE_HT, dict_ht_open,
    DICT_TYPE_UNIX, dict_unix_open,
    DICT_TYPE_TCP, dict_tcp_open,
#ifdef HAS_DBM
    DICT_TYPE_DBM, dict_dbm_open,
#endif
#ifdef HAS_DB
    DICT_TYPE_HASH, dict_hash_open,
    DICT_TYPE_BTREE, dict_btree_open,
#endif
#ifdef HAS_NIS
    DICT_TYPE_NIS, dict_nis_open,
#endif
#ifdef HAS_NISPLUS
    DICT_TYPE_NISPLUS, dict_nisplus_open,
#endif
#ifdef HAS_NETINFO
    DICT_TYPE_NETINFO, dict_ni_open,
#endif
#ifdef HAS_POSIX_REGEXP
    DICT_TYPE_REGEXP, dict_regexp_open,
#endif
    DICT_TYPE_STATIC, dict_static_open,
    DICT_TYPE_CIDR, dict_cidr_open,
    DICT_TYPE_THASH, dict_thash_open,
    DICT_TYPE_SOCKMAP, dict_sockmap_open,
    DICT_TYPE_FAIL, dict_fail_open,
    DICT_TYPE_PIPE, dict_pipe_open,
    DICT_TYPE_RANDOM, dict_random_open,
    DICT_TYPE_UNION, dict_union_open,
    DICT_TYPE_INLINE, dict_inline_open,
#ifndef USE_DYNAMIC_MAPS
#ifdef HAS_PCRE
    DICT_TYPE_PCRE, dict_pcre_open,
#endif
#ifdef HAS_CDB
    DICT_TYPE_CDB, dict_cdb_open,
#endif
#ifdef HAS_SDBM
    DICT_TYPE_SDBM, dict_sdbm_open,
#endif
#ifdef HAS_LMDB
    DICT_TYPE_LMDB, dict_lmdb_open,
#endif
#endif					/* !USE_DYNAMIC_MAPS */
    0,
};

static HTABLE *dict_open_hash;

 /*
  * Extension hooks.
  */
static DICT_OPEN_EXTEND_FN dict_open_extend_hook;
static DICT_MAPNAMES_EXTEND_FN dict_mapnames_extend_hook;

 /*
  * Workaround.
  */
DEFINE_DICT_LMDB_MAP_SIZE;
DEFINE_DICT_DB_CACHE_SIZE;

/* dict_open_init - one-off initialization */

static void dict_open_init(void)
{
    const char *myname = "dict_open_init";
    const DICT_OPEN_INFO *dp;

    if (dict_open_hash != 0)
	msg_panic("%s: multiple initialization", myname);
    dict_open_hash = htable_create(10);

    for (dp = dict_open_info; dp->type; dp++)
	htable_enter(dict_open_hash, dp->type, (void *) dp);
}

/* dict_open - open dictionary */

DICT   *dict_open(const char *dict_spec, int open_flags, int dict_flags)
{
    char   *saved_dict_spec = mystrdup(dict_spec);
    char   *dict_name;
    DICT   *dict;

    if ((dict_name = split_at(saved_dict_spec, ':')) == 0)
	msg_fatal("open dictionary: expecting \"type:name\" form instead of \"%s\"",
		  dict_spec);

    dict = dict_open3(saved_dict_spec, dict_name, open_flags, dict_flags);
    myfree(saved_dict_spec);
    return (dict);
}


/* dict_open3 - open dictionary */

DICT   *dict_open3(const char *dict_type, const char *dict_name,
		           int open_flags, int dict_flags)
{
    const char *myname = "dict_open";
    DICT_OPEN_INFO *dp;
    DICT_OPEN_FN open_fn;
    DICT   *dict;

    if (*dict_type == 0 || *dict_name == 0)
	msg_fatal("open dictionary: expecting \"type:name\" form instead of \"%s:%s\"",
		  dict_type, dict_name);
    if (dict_open_hash == 0)
	dict_open_init();
    if ((dp = (DICT_OPEN_INFO *) htable_find(dict_open_hash, dict_type)) == 0) {
	if (dict_open_extend_hook != 0
	    && (open_fn = dict_open_extend_hook(dict_type)) != 0) {
	    dict_open_register(dict_type, open_fn);
	    dp = (DICT_OPEN_INFO *) htable_find(dict_open_hash, dict_type);
	}
	if (dp == 0)
	    return (dict_surrogate(dict_type, dict_name, open_flags, dict_flags,
			     "unsupported dictionary type: %s", dict_type));
    }
    if ((dict = dp->open(dict_name, open_flags, dict_flags)) == 0)
	return (dict_surrogate(dict_type, dict_name, open_flags, dict_flags,
			    "cannot open %s:%s: %m", dict_type, dict_name));
    if (msg_verbose)
	msg_info("%s: %s:%s", myname, dict_type, dict_name);
    /* XXX The choice between wait-for-lock or no-wait is hard-coded. */
    if (dict->flags & DICT_FLAG_OPEN_LOCK) {
	if (dict->flags & DICT_FLAG_LOCK)
	    msg_panic("%s: attempt to open %s:%s with both \"open\" lock and \"access\" lock",
		      myname, dict_type, dict_name);
	/* Multi-writer safe map: downgrade persistent lock to temporary. */
	if (dict->flags & DICT_FLAG_MULTI_WRITER) {
	    dict->flags &= ~DICT_FLAG_OPEN_LOCK;
	    dict->flags |= DICT_FLAG_LOCK;
	}
	/* Multi-writer unsafe map: acquire exclusive lock or bust. */
	else if (dict->lock(dict, MYFLOCK_OP_EXCLUSIVE | MYFLOCK_OP_NOWAIT) < 0)
	    msg_fatal("%s:%s: unable to get exclusive lock: %m",
		      dict_type, dict_name);
    }
    /* Last step: insert proxy for UTF-8 syntax checks and casefolding. */
    if ((dict->flags & DICT_FLAG_UTF8_ACTIVE) == 0
	&& DICT_NEED_UTF8_ACTIVATION(util_utf8_enable, dict_flags))
	dict = dict_utf8_activate(dict);
    return (dict);
}

/* dict_open_register - register dictionary type */

void    dict_open_register(const char *type, DICT_OPEN_FN open)
{
    const char *myname = "dict_open_register";
    DICT_OPEN_INFO *dp;
    HTABLE_INFO *ht;

    if (dict_open_hash == 0)
	dict_open_init();
    if (htable_find(dict_open_hash, type))
	msg_panic("%s: dictionary type exists: %s", myname, type);
    dp = (DICT_OPEN_INFO *) mymalloc(sizeof(*dp));
    dp->open = open;
    ht = htable_enter(dict_open_hash, type, (void *) dp);
    dp->type = ht->key;
}

/* dict_open_extend - register alternate dictionary search routine */

DICT_OPEN_EXTEND_FN dict_open_extend(DICT_OPEN_EXTEND_FN new_cb)
{
    DICT_OPEN_EXTEND_FN old_cb;

    old_cb = dict_open_extend_hook;
    dict_open_extend_hook = new_cb;
    return (old_cb);
}

/* dict_sort_alpha_cpp - qsort() callback */

static int dict_sort_alpha_cpp(const void *a, const void *b)
{
    return (strcmp(((char **) a)[0], ((char **) b)[0]));
}

/* dict_mapnames - return an ARGV of available map_names */

ARGV   *dict_mapnames()
{
    HTABLE_INFO **ht_info;
    HTABLE_INFO **ht;
    DICT_OPEN_INFO *dp;
    ARGV   *mapnames;

    if (dict_open_hash == 0)
	dict_open_init();
    mapnames = argv_alloc(dict_open_hash->used + 1);
    for (ht_info = ht = htable_list(dict_open_hash); *ht; ht++) {
	dp = (DICT_OPEN_INFO *) ht[0]->value;
	argv_add(mapnames, dp->type, ARGV_END);
    }
    if (dict_mapnames_extend_hook != 0)
	(void) dict_mapnames_extend_hook(mapnames);
    qsort((void *) mapnames->argv, mapnames->argc, sizeof(mapnames->argv[0]),
	  dict_sort_alpha_cpp);
    myfree((void *) ht_info);
    argv_terminate(mapnames);
    return mapnames;
}

/* dict_mapnames_extend - register alternate dictionary type list routine */

DICT_MAPNAMES_EXTEND_FN dict_mapnames_extend(DICT_MAPNAMES_EXTEND_FN new_cb)
{
    DICT_MAPNAMES_EXTEND_FN old_cb;

    old_cb = dict_mapnames_extend_hook;
    dict_mapnames_extend_hook = new_cb;
    return (old_cb);
}

/* dict_type_override - disguise a dictionary type */

void    dict_type_override(DICT *dict, const char *type)
{
    myfree(dict->type);
    dict->type = mystrdup(type);
}

#ifdef TEST

 /*
  * Proof-of-concept test program.
  */
int     main(int argc, char **argv)
{
    dict_test(argc, argv);
    return (0);
}

#endif
