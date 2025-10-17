/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 20, 2021.
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

#include <dict.h>
#include <msg.h>
#include <mymalloc.h>
#include <stringops.h>
#include <dynamicmaps.h>

/* Global library. */

#include <dict_proxy.h>
#include <dict_ldap.h>
#include <dict_mysql.h>
#include <dict_pgsql.h>
#include <dict_sqlite.h>
#include <dict_memcache.h>
#include <mail_dict.h>
#include <mail_params.h>
#include <mail_dict.h>

typedef struct {
    char   *type;
    struct DICT *(*open) (const char *, int, int);
} DICT_OPEN_INFO;

static const DICT_OPEN_INFO dict_open_info[] = {
    DICT_TYPE_PROXY, dict_proxy_open,
#ifndef USE_DYNAMIC_MAPS
#ifdef HAS_LDAP
    DICT_TYPE_LDAP, dict_ldap_open,
#endif
#ifdef HAS_MYSQL
    DICT_TYPE_MYSQL, dict_mysql_open,
#endif
#ifdef HAS_PGSQL
    DICT_TYPE_PGSQL, dict_pgsql_open,
#endif
#ifdef HAS_SQLITE
    DICT_TYPE_SQLITE, dict_sqlite_open,
#endif
#endif					/* !USE_DYNAMIC_MAPS */
    DICT_TYPE_MEMCACHE, dict_memcache_open,
    0,
};

/* mail_dict_init - dictionaries that depend on Postfix-specific interfaces */

void    mail_dict_init(void)
{
    const DICT_OPEN_INFO *dp;

#ifdef USE_DYNAMIC_MAPS
    char   *path;

    path = concatenate(var_meta_dir, "/", "dynamicmaps.cf",
#ifdef SHLIB_VERSION
		       ".", SHLIB_VERSION,
#endif
		       (char *) 0);
    dymap_init(path, var_shlib_dir);
    myfree(path);
#endif

    for (dp = dict_open_info; dp->type; dp++)
	dict_open_register(dp->type, dp->open);
}

#ifdef TEST

 /*
  * Proof-of-concept test program.
  */

#include <mail_proto.h>
#include <mail_params.h>

int     main(int argc, char **argv)
{
    var_queue_dir = DEF_QUEUE_DIR;
    var_proxymap_service = DEF_PROXYMAP_SERVICE;
    var_proxywrite_service = DEF_PROXYWRITE_SERVICE;
    var_ipc_timeout = 3600;
    mail_dict_init();
    dict_test(argc, argv);
    return (0);
}

#endif
