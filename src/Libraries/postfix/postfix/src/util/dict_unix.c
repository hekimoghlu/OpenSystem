/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 8, 2022.
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
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <pwd.h>
#include <grp.h>

/* Utility library. */

#include "msg.h"
#include "mymalloc.h"
#include "vstring.h"
#include "stringops.h"
#include "dict.h"
#include "dict_unix.h"

/* Application-specific. */

typedef struct {
    DICT    dict;			/* generic members */
} DICT_UNIX;

/* dict_unix_getpwnam - find password table entry */

static const char *dict_unix_getpwnam(DICT *dict, const char *key)
{
    struct passwd *pwd;
    static VSTRING *buf;
    static int sanity_checked;

    dict->error = 0;

    /*
     * Optionally fold the key.
     */
    if (dict->flags & DICT_FLAG_FOLD_FIX) {
	if (dict->fold_buf == 0)
	    dict->fold_buf = vstring_alloc(10);
	vstring_strcpy(dict->fold_buf, key);
	key = lowercase(vstring_str(dict->fold_buf));
    }
    if ((pwd = getpwnam(key)) == 0) {
	if (sanity_checked == 0) {
	    sanity_checked = 1;
	    errno = 0;
	    if (getpwuid(0) == 0) {
		msg_warn("cannot access UNIX password database: %m");
		dict->error = DICT_ERR_RETRY;
	    }
	}
	return (0);
    } else {
	if (buf == 0)
	    buf = vstring_alloc(10);
	sanity_checked = 1;
	vstring_sprintf(buf, "%s:%s:%ld:%ld:%s:%s:%s",
			pwd->pw_name, pwd->pw_passwd, (long) pwd->pw_uid,
			(long) pwd->pw_gid, pwd->pw_gecos, pwd->pw_dir,
			pwd->pw_shell);
	return (vstring_str(buf));
    }
}

/* dict_unix_getgrnam - find group table entry */

static const char *dict_unix_getgrnam(DICT *dict, const char *key)
{
    struct group *grp;
    static VSTRING *buf;
    char  **cpp;
    static int sanity_checked;

    dict->error = 0;

    /*
     * Optionally fold the key.
     */
    if (dict->flags & DICT_FLAG_FOLD_FIX) {
	if (dict->fold_buf == 0)
	    dict->fold_buf = vstring_alloc(10);
	vstring_strcpy(dict->fold_buf, key);
	key = lowercase(vstring_str(dict->fold_buf));
    }
    if ((grp = getgrnam(key)) == 0) {
	if (sanity_checked == 0) {
	    sanity_checked = 1;
	    errno = 0;
	    if (getgrgid(0) == 0) {
		msg_warn("cannot access UNIX group database: %m");
		dict->error = DICT_ERR_RETRY;
	    }
	}
	return (0);
    } else {
	if (buf == 0)
	    buf = vstring_alloc(10);
	sanity_checked = 1;
	vstring_sprintf(buf, "%s:%s:%ld:",
			grp->gr_name, grp->gr_passwd, (long) grp->gr_gid);
	for (cpp = grp->gr_mem; *cpp; cpp++) {
	    vstring_strcat(buf, *cpp);
	    if (cpp[1])
		VSTRING_ADDCH(buf, ',');
	}
	VSTRING_TERMINATE(buf);
	return (vstring_str(buf));
    }
}

/* dict_unix_close - close UNIX map */

static void dict_unix_close(DICT *dict)
{
    if (dict->fold_buf)
	vstring_free(dict->fold_buf);
    dict_free(dict);
}

/* dict_unix_open - open UNIX map */

DICT   *dict_unix_open(const char *map, int open_flags, int dict_flags)
{
    DICT_UNIX *dict_unix;
    struct dict_unix_lookup {
	char   *name;
	const char *(*lookup) (DICT *, const char *);
    };
    static struct dict_unix_lookup dict_unix_lookup[] = {
	"passwd.byname", dict_unix_getpwnam,
	"group.byname", dict_unix_getgrnam,
	0,
    };
    struct dict_unix_lookup *lp;

    /*
     * Sanity checks.
     */
    if (open_flags != O_RDONLY)
	return (dict_surrogate(DICT_TYPE_UNIX, map, open_flags, dict_flags,
			       "%s:%s map requires O_RDONLY access mode",
			       DICT_TYPE_UNIX, map));

    /*
     * "Open" the database.
     */
    for (lp = dict_unix_lookup; /* void */ ; lp++) {
	if (lp->name == 0)
	    return (dict_surrogate(DICT_TYPE_UNIX, map, open_flags, dict_flags,
			      "unknown table: %s:%s", DICT_TYPE_UNIX, map));
	if (strcmp(map, lp->name) == 0)
	    break;
    }
    dict_unix = (DICT_UNIX *) dict_alloc(DICT_TYPE_UNIX, map,
					 sizeof(*dict_unix));
    dict_unix->dict.lookup = lp->lookup;
    dict_unix->dict.close = dict_unix_close;
    dict_unix->dict.flags = dict_flags | DICT_FLAG_FIXED;
    if (dict_flags & DICT_FLAG_FOLD_FIX)
	dict_unix->dict.fold_buf = vstring_alloc(10);
    dict_unix->dict.owner.status = DICT_OWNER_TRUSTED;

    return (DICT_DEBUG (&dict_unix->dict));
}
