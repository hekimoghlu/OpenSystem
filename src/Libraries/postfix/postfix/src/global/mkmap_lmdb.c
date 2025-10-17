/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 13, 2023.
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
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>

/* Utility library. */

#include <msg.h>
#include <mymalloc.h>
#include <stringops.h>
#include <dict.h>
#include <dict_lmdb.h>
#include <myflock.h>
#include <warn_stat.h>

#ifdef HAS_LMDB
#ifdef PATH_LMDB_H
#include PATH_LMDB_H
#else
#include <lmdb.h>
#endif

/* Global library. */

#include <mail_conf.h>
#include <mail_params.h>

/* Application-specific. */

#include "mkmap.h"

/* mkmap_lmdb_open */

MKMAP  *mkmap_lmdb_open(const char *path)
{
    MKMAP  *mkmap = (MKMAP *) mymalloc(sizeof(*mkmap));

    /*
     * Fill in the generic members.
     */
    mkmap->open = dict_lmdb_open;
    mkmap->after_open = 0;
    mkmap->after_close = 0;

    /*
     * LMDB uses MVCC so it needs no special lock management here.
     */

    return (mkmap);
}

#endif
