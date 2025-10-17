/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 13, 2025.
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
#include <unistd.h>

/* Utility library. */

#include <msg.h>
#include <mymalloc.h>
#include <stringops.h>
#include <dict.h>
#include <dict_sdbm.h>
#include <myflock.h>

/* Application-specific. */

#include "mkmap.h"

#ifdef HAS_SDBM

#include <sdbm.h>

typedef struct MKMAP_SDBM {
    MKMAP   mkmap;			/* parent class */
    char   *lock_file;			/* path name */
    int     lock_fd;			/* -1 or open locked file */
} MKMAP_SDBM;

/* mkmap_sdbm_after_close - clean up after closing database */

static void mkmap_sdbm_after_close(MKMAP *mp)
{
    MKMAP_SDBM *mkmap = (MKMAP_SDBM *) mp;

    if (mkmap->lock_fd >= 0 && close(mkmap->lock_fd) < 0)
	msg_warn("close %s: %m", mkmap->lock_file);
    myfree(mkmap->lock_file);
}

/* mkmap_sdbm_open - create or open database */

MKMAP  *mkmap_sdbm_open(const char *path)
{
    MKMAP_SDBM *mkmap = (MKMAP_SDBM *) mymalloc(sizeof(*mkmap));
    char   *pag_file;
    int     pag_fd;

    /*
     * Fill in the generic members.
     */
    mkmap->lock_file = concatenate(path, ".dir", (char *) 0);
    mkmap->mkmap.open = dict_sdbm_open;
    mkmap->mkmap.after_open = 0;
    mkmap->mkmap.after_close = mkmap_sdbm_after_close;

    /*
     * Unfortunately, not all systems support locking on open(), so we open
     * the .dir and .pag files before truncating them. Keep one file open for
     * locking.
     */
    if ((mkmap->lock_fd = open(mkmap->lock_file, O_CREAT | O_RDWR, 0644)) < 0)
	msg_fatal("open %s: %m", mkmap->lock_file);

    pag_file = concatenate(path, ".pag", (char *) 0);
    if ((pag_fd = open(pag_file, O_CREAT | O_RDWR, 0644)) < 0)
	msg_fatal("open %s: %m", pag_file);
    if (close(pag_fd))
	msg_warn("close %s: %m", pag_file);
    myfree(pag_file);

    /*
     * Get an exclusive lock - we're going to change the database so we can't
     * have any spectators.
     */
    if (myflock(mkmap->lock_fd, INTERNAL_LOCK, MYFLOCK_OP_EXCLUSIVE) < 0)
	msg_fatal("lock %s: %m", mkmap->lock_file);

    return (&mkmap->mkmap);
}

#endif
